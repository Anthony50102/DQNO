import matplotlib.pyplot as plt
import torch
from dqno.models.dqfno import DQFNO
from dqno.losses.custom_losses import MultiTaskLoss
from dqno.losses.data_losses import LpLoss, H1Loss
import os
import json
from datetime import datetime

def create_run_directory(config, base_dir='./runs'):
    """
    Creates a unique directory for the current training run and saves meta data.

    Parameters:
        config (dict or object): The training configuration containing hyperparameters 
                                 and other relevant settings. If not a dict, an attempt 
                                 is made to convert it via __dict__.
        base_dir (str): The base folder where run directories will be created (default: './runs').

    Returns:
        str: The path to the newly created run directory.
    """
    # Ensure the base directory exists.
    os.makedirs(base_dir, exist_ok=True)
    
    # Create a unique run ID based on the current timestamp.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=False)
    
    # Convert config to a dictionary if it isn't already.
    if isinstance(config, dict):
        config_to_save = config
    else:
        try:
            config_to_save = config.__dict__
        except Exception:
            config_to_save = str(config)
    
    # Save the configuration as a JSON file.
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=4, default=str)
    
    # Save additional meta data (e.g., run start time, run ID).
    meta_data = {
        "run_id": run_id,
        "start_time": datetime.now().isoformat(),
        "config_file": config_path,
    }
    meta_path = os.path.join(run_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=4)
    
    return run_dir

def initialize_model(config, dx, device):
    model = DQFNO(
        modes=config.dqfno.modes,
        in_channels=config.dqfno.data_channels,
        out_channels=config.dqfno.data_channels,
        hidden_channels=config.dqfno.hidden_channels,
        n_layers=config.dqfno.n_layers,
        dx=dx,
        derived_type=config.dqfno.derived_type,
    )
    model.to(device)
    return model

def get_loss_object(config):
    selector_state = lambda y_pred, y: (y_pred[0], y[0])
    selector_derived = lambda y_pred, y: (y_pred[1], y[1])
    
    losses = []
    selectors = []
    for loss, weight in zip(config.losses.losses, config.losses.weights):
        if loss == 'lp':
            losses.append(LpLoss(d=4, p=2, reduction='mean'))
            selectors.append(selector_state)
        elif loss == 'h1':
            losses.append(H1Loss(d=2))
            selectors.append(selector_state)
        elif loss == 'derived':
            losses.append(torch.nn.MSELoss())
            selectors.append(selector_derived)
    
    return MultiTaskLoss(
        loss_functions=losses,
        scales=config.losses.weights,
        multi_output=True,
        input_selectors=selectors
    )

def plot_and_save_loss(train_losses, test_losses, n_epochs, run_dir):
    plt.figure()
    plt.plot(range(n_epochs), train_losses, label='Train Loss')
    plt.plot(range(n_epochs), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train vs Test Loss')
    plt.savefig(os.path.join(run_dir, 'loss_plot.png'))
    plt.close()
