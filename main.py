import argparse
import yaml
import torch
from pathlib import Path

# Import modules from your 'src' package
from src import data_setup, model_builder, engine, utils


def main(config_path: str):
    # 1. Load configuration file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_config = config['data_params']

    # 2. LOOP THROUGH MODELS IN CONFIG AND TRAIN
    print("\n[INFO] Starting PHASE 1: Model Comparison")
    for model_conf in config['model_configs']:
        model_name = model_conf['name']
        print(f"\n--- Training model: {model_name} ---")

        # 2.1. Build the model
        model, auto_transforms = model_builder.create_model(
            model_name=model_name,
            num_classes=data_config['num_classes']
        )

        model = model.to(device)

        train_dataloader, test_dataloader, _ = data_setup.create_dataloaders(
            train_dir=data_config['train_dir'],
            test_dir=data_config['test_dir'],
            transform=auto_transforms,  # Pass in created transform
            batch_size=data_config['batch_size']
        )

        ### NEW: Flexible Loss Function Initialization ###
        # Get loss function name from config file
        loss_fn_name = config['training_params']['loss_function']
        # Use getattr to get the loss class from torch.nn
        loss_fn = getattr(torch.nn, loss_fn_name)()

        ### NEW: Flexible Optimizer Initialization ###
        optimizer_config = config['training_params']['optimizer']
        # Get optimizer name from config file
        optimizer_name = optimizer_config['name']
        # Use getattr to get optimizer class from torch.optim
        optimizer_class = getattr(torch.optim, optimizer_name)
        # Initialize optimizer with parameters from config
        optimizer = optimizer_class(params=model.parameters(),
                                    lr=optimizer_config['lr'])

        # 2.2. Train the model
        results = engine.train(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,  # Pass optimizer object
            loss_fn=loss_fn,      # Pass loss_fn object
            epochs=config['training_params']['epochs'],
            device=device,
            # Log to TensorBoard
            writer=utils.create_writer(
                experiment_name="phase1", model_name=model_name)
        )

        # 3. SAVE THE CURRENT MODEL'S RESULTS TO FILE
        utils.save_results(results=results,
                           model_name=model_name,
                           target_dir="results")  # Create the "results" directory to store the file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train and compare models.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the .yaml configuration file.")
    args = parser.parse_args()
    main(config_path=args.config)
