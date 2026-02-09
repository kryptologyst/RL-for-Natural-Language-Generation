"""Utility functions for the RL text generation project."""

import torch
import numpy as np
import random
from typing import Optional, Dict, Any, List
import os
import json
from datetime import datetime


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification ("auto", "cpu", "cuda", "mps")
        
    Returns:
        PyTorch device
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def save_config(config: Dict[str, Any], filepath: str) -> None:
    """Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2, default=str)


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        filepath: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def create_log_dir(base_dir: str = "logs") -> str:
    """Create a unique log directory with timestamp.
    
    Args:
        base_dir: Base directory for logs
        
    Returns:
        Path to created log directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> tuple:
    """Compute confidence interval for data.
    
    Args:
        data: List of values
        confidence: Confidence level (default: 0.95)
        
    Returns:
        Tuple of (mean, margin_of_error)
    """
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)
    
    # Z-score for confidence level
    z_score = 1.96 if confidence == 0.95 else 2.576  # 99% confidence
    
    margin_of_error = z_score * (std / np.sqrt(n))
    
    return mean, margin_of_error


def moving_average(data: List[float], window_size: int = 10) -> List[float]:
    """Compute moving average of data.
    
    Args:
        data: List of values
        window_size: Size of moving window
        
    Returns:
        List of moving averages
    """
    if len(data) < window_size:
        return data
    
    moving_avg = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        window_data = data[start_idx:i + 1]
        moving_avg.append(np.mean(window_data))
    
    return moving_avg


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: torch.nn.Module, input_shape: tuple = None) -> None:
    """Print model information.
    
    Args:
        model: PyTorch model
        input_shape: Input shape for the model
    """
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {count_parameters(model):,}")
    
    if input_shape:
        try:
            dummy_input = torch.randn(1, *input_shape)
            output = model(dummy_input)
            print(f"Input shape: {input_shape}")
            print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"Could not determine output shape: {e}")


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score
            model: Model to potentially restore weights
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        
        return False
    
    def save_checkpoint(self, model: torch.nn.Module) -> None:
        """Save model checkpoint."""
        self.best_weights = model.state_dict().copy()


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if configuration is valid
    """
    required_keys = ['env', 'rl', 'model', 'training']
    
    for key in required_keys:
        if key not in config:
            print(f"Missing required config key: {key}")
            return False
    
    # Validate environment config
    env_config = config['env']
    required_env_keys = ['vocab_size', 'max_length', 'reward_type']
    for key in required_env_keys:
        if key not in env_config:
            print(f"Missing required env config key: {key}")
            return False
    
    # Validate RL config
    rl_config = config['rl']
    required_rl_keys = ['algorithm', 'learning_rate', 'gamma']
    for key in required_rl_keys:
        if key not in rl_config:
            print(f"Missing required rl config key: {key}")
            return False
    
    return True


def create_experiment_name(config: Dict[str, Any]) -> str:
    """Create a descriptive experiment name from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Experiment name string
    """
    algorithm = config['rl']['algorithm']
    model_type = config['model']['model_type']
    reward_type = config['env']['reward_type']
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    return f"{algorithm}_{model_type}_{reward_type}_{timestamp}"
