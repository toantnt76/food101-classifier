
"""
Contains various utility functions for PyTorch model training and saving.
"""
import matplotlib.patches as patches  # NEW: Import the patches module
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import time
from PIL import Image
from IPython.display import display
from adjustText import adjust_text


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)


def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str = None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    from datetime import datetime
    import os

    # Get timestamp of current date (all experiments on certain day live in same folder)
    # returns current date in YYYY-MM-DD format
    timestamp = datetime.now().strftime("%Y-%m-%d")

    if extra:
        # Create log directory path
        log_dir = os.path.join(
            "runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)


def save_results(results: dict,
                 model_name: str,
                 target_dir: str):
    """Save the training results dictionary into a CSV file.

    Args:
        results (dict): Dictionary containing loss and accuracy history.
        model_name (str): Name of the model to use for naming the file.
        target_dir (str): Directory where the results file will be saved.
    """
    # Create the directory if it does not exist
    os.makedirs(target_dir, exist_ok=True)

    # Create a DataFrame from the dictionary
    results_df = pd.DataFrame(results)

    # Add an epoch column for easier plotting
    results_df['epoch'] = range(len(results_df))

    # Define the file name
    file_name = f"{model_name}_phase1_results.csv"
    save_path = os.path.join(target_dir, file_name)

    # Save the DataFrame to a CSV file
    print(f"[INFO] Saving results to: {save_path}")
    results_df.to_csv(save_path, index=False)


def calculate_model_size(model_name: str, num_classes: int) -> dict:
    """Initializes a model and calculates its size based on parameters."""
    # Initialize the model architecture (no training needed)
    from . import model_builder
    model, _ = model_builder.create_model(model_name=model_name,
                                          num_classes=num_classes)

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Calculate size in MB (1 float32 param = 4 bytes)
    model_size_mb = (total_params * 4) / (1024**2)

    return {"total_params": total_params, "size_mb": model_size_mb}


def benchmark_inference_speed(model_name: str, num_classes: int, device: str = "cpu", iterations: int = 100) -> float:
    """Measures the average inference time of a model."""
    # Initialize the model and move it to the target device
    from . import model_builder
    model, transform = model_builder.create_model(model_name=model_name,
                                                  num_classes=num_classes)
    model.to(device)

    # CRITICAL: Set the model to evaluation mode
    model.eval()

    # Create a dummy PIL image first, then let the full transform pipeline
    # process it. This guarantees the final tensor is exactly what the model expects,
    # including potential center cropping (like in ViT).

    # The initial size (300, 300) is arbitrary, as it will be transformed.
    dummy_pil_image = Image.new('RGB', (300, 300))

    # Apply the entire transform pipeline to the dummy image to get the correct tensor shape,
    # then add a batch dimension with unsqueeze(0).
    dummy_input = transform(dummy_pil_image).unsqueeze(0).to(device)

    # Warm-up runs (to initialize CUDA, etc., not included in timing)
    with torch.inference_mode():
        for _ in range(10):
            _ = model(dummy_input)

    # Actual benchmark loop
    start_time = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iterations):
            _ = model(dummy_input)
    end_time = time.perf_counter()

    # Calculate average time in milliseconds (ms)
    avg_time_ms = ((end_time - start_time) / iterations) * 1000

    return avg_time_ms


def analyze_and_visualize_efficiency(model_names: list,
                                     num_classes: int,
                                     device: str = "cpu"):
    """
    Performs a full efficiency analysis (size and speed) for a list of models,
    then visualizes the results with bar charts and a styled summary table.

    Args:
        model_names (list): A list of model names to analyze.
        num_classes (int): The number of output classes for the models.
        device (str): The device to benchmark inference speed on (e.g., "cpu").
    """
    # --- 1. Run the Analysis & Collect Results ---
    print(f"--- Running Efficiency Analysis on device: '{device}' ---")

    efficiency_results = []
    for name in model_names:
        print(f"Analyzing model: {name}...")
        # Calculate size using the imported function
        size_info = calculate_model_size(
            model_name=name, num_classes=num_classes)
        # Benchmark speed on CPU using the imported function
        inference_time_ms = benchmark_inference_speed(model_name=name,
                                                      num_classes=num_classes,
                                                      device=device)  # Pass 'cpu' here
        efficiency_results.append({
            "Model": name,
            "Parameters (M)": size_info["total_params"] / 1_000_000,
            "Size (MB)": size_info["size_mb"],
            "Avg Inference (ms) on CPU": inference_time_ms
        })

    efficiency_df = pd.DataFrame(efficiency_results).set_index("Model")
    # --- 2. Visualize the Efficiency Analysis ---
    # Sort the DataFrame by Size for a cleaner plot
    df_sorted_by_size = efficiency_df.sort_values("Size (MB)", ascending=False)
    df_sorted_by_speed = efficiency_df.sort_values(
        "Avg Inference (ms) on CPU", ascending=False)

    # Create a figure with two subplots, side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 2.1 Model Size Comparison
    bars1 = ax1.barh(df_sorted_by_size.index,
                     df_sorted_by_size["Size (MB)"], color='skyblue')
    ax1.set_title("Model Size Comparison", fontsize=16)
    ax1.set_xlabel("Size (MB)", fontsize=12)
    ax1.set_ylabel("")
    ax1.grid(axis='x', linestyle='--', alpha=0.7)

    # Add value labels to the bars
    ax1.bar_label(bars1, fmt='%.2f MB', padding=3)
    # Add a 15% padding to the right of the longest bar
    ax1.set_xlim(right=df_sorted_by_size["Size (MB)"].max() * 1.15)

    # 2.2 Inference Speed Comparison (on CPU)
    bars2 = ax2.barh(df_sorted_by_speed.index,
                     df_sorted_by_speed["Avg Inference (ms) on CPU"], color='salmon')
    ax2.set_title("Inference Speed Comparison (on CPU)", fontsize=16)
    ax2.set_xlabel("Average Inference Time (ms)", fontsize=12)
    ax2.set_ylabel("")
    ax2.grid(axis='x', linestyle='--', alpha=0.7)

    # Add value labels to the bars
    ax2.bar_label(bars2, fmt='%.2f ms', padding=3)
    # Add a 15% padding to the right of the longest bar
    ax2.set_xlim(
        right=df_sorted_by_size["Avg Inference (ms) on CPU"].max() * 1.15)

    # CRITICAL: Add the 30 FPS threshold line
    FPS_THRESHOLD_MS = 1000 / 30
    ax2.axvline(x=FPS_THRESHOLD_MS, color='red', linestyle='--',
                linewidth=2, label=f'> 30 FPS Threshold ({FPS_THRESHOLD_MS:.2f}ms)')
    ax2.legend()

    # --- Final Touches ---
    fig.suptitle("Efficiency Analysis: Size vs. Speed", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- 3. Create and Display a Styled DataFrame ---
    print("\n" + "="*50)
    print("Efficiency Analysis Summary")
    print("="*50)

    # 3.1 Define our custom highlighting function
    # FPS threshold (30 FPS is approximately 33.33ms per frame)
    FPS_THRESHOLD_MS = 1000 / 30

    def highlight_fast_inference(series):
        """
        Highlights inference times that are BELOW the threshold (i.e., > 30 FPS).
        """
        return ['background-color: lightgreen' if val < FPS_THRESHOLD_MS else '' for val in series]

    # 3.2 Apply the single styling rule to our DataFrame
    styled_df = efficiency_df.style.apply(
        highlight_fast_inference,
        axis=0,
        # Apply to this column ONLY
        subset=['Avg Inference (ms) on CPU']
    ).format(
        "{:.2f}"  # Format all numbers to 2 decimal places
    )

    # 3.3 Display the final styled DataFrame
    display(styled_df)
    # NEW: Return the DataFrame for further use (e.g., creating the final decision matrix)
    return efficiency_df


def plot_tradeoff_scatter(decision_df: pd.DataFrame, acc_threshold=80, speed_threshold_ms=33.33):
    """
    Creates a final, highly polished trade-off scatter plot.
    Features a correctly centered and rotated "Optimal Zone" label.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    SCALING_FACTOR = 20

    colors = plt.cm.get_cmap('tab10', len(decision_df))
    color_map = {model_name: colors(
        i) for i, model_name in enumerate(decision_df.index)}

    scatter = ax.scatter(
        x=decision_df["Avg Inference (ms) on CPU"],
        y=decision_df["Best Test Acc (%)"],
        s=decision_df["Size (MB)"] * SCALING_FACTOR,
        c=decision_df.index.map(color_map),
        alpha=0.7,
        zorder=5
    )

    # Get the final plot limits AFTER drawing the scatter points
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # --- CHANGE: Correctly anchor, shade, AND CENTER the "Optimal Zone" text ---

    # Define the rectangle for the optimal zone
    optimal_zone = patches.Rectangle(
        (x_min, acc_threshold),
        speed_threshold_ms - x_min,
        y_max - acc_threshold,
        facecolor='lightgray',
        alpha=0.2,  # Made it slightly lighter for better text readability
        zorder=0
    )
    ax.add_patch(optimal_zone)

    # Calculate the precise center of the shaded zone
    center_x = (x_min + speed_threshold_ms) / 2
    center_y = (acc_threshold + y_max) / 2

    # Add rotated text annotation precisely in the center
    ax.text(
        center_x,  # Use the calculated center x
        center_y,  # Use the calculated center y
        'Optimal Zone',
        fontsize=18,
        fontweight='bold',
        color='gray',
        ha='center',  # Horizontal alignment to the center
        va='center',  # Vertical alignment to the center
        alpha=0.6,
        rotation='vertical',
        zorder=1
    )
    # --- END OF CHANGE ---

    # Add model name annotations
    texts = []
    for i, row in decision_df.iterrows():
        texts.append(ax.text(row["Avg Inference (ms) on CPU"],
                             row["Best Test Acc (%)"] + 0.1,
                             i, fontsize=12))
    adjust_text(texts, ax=ax)

    # Add threshold lines
    ax.axvline(x=speed_threshold_ms, color='grey', linestyle='--',
               label=f'Speed Threshold ({speed_threshold_ms:.1f}ms)')
    ax.axhline(y=acc_threshold, color='grey', linestyle='--',
               label=f'Accuracy Threshold ({acc_threshold}%)')

    threshold_legend = ax.legend(loc="upper right", fontsize=12)
    ax.add_artist(threshold_legend)

    # Create the custom size legend
    df_sorted_by_size = decision_df.sort_values("Size (MB)", ascending=True)
    legend_handles = []
    LEGEND_MARKER_SIZE = 100
    for index, row in df_sorted_by_size.iterrows():
        handle = ax.scatter([], [], s=LEGEND_MARKER_SIZE, c=[color_map[index]],
                            alpha=0.6, label=f'{row["Size (MB)"]:.2f} MB ({index})')
        legend_handles.append(handle)
    size_legend = ax.legend(handles=legend_handles, loc="lower right",
                            title="Model Size (Actual)", fontsize=12, title_fontsize=13)

    # Aesthetics and labels
    ax.set_title("Model Trade-off: Speed vs. Accuracy vs. Size", fontsize=20)
    ax.set_xlabel(
        "Inference Time per Image (ms) on CPU - Slower is Better <-", fontsize=14)
    ax.set_ylabel("Test Accuracy (%) - Higher is Better ->", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.6, zorder=-1)

    # Reset the limits to ensure the plot is tight after drawing everything
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.show()
