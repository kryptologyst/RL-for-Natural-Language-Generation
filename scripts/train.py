#!/usr/bin/env python3
"""Main training script for RL-based Natural Language Generation.

This script provides a command-line interface for training RL agents
on text generation tasks with various configurations.
"""

import argparse
import yaml
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.environments.nlg_env import NLGConfig
from src.algorithms.rl_agents import RLConfig
from src.training.trainer import RLTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_configs(config: dict) -> tuple:
    """Create configuration objects from dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (env_config, rl_config, model_config, training_config)
    """
    # Environment config
    env_config = NLGConfig(**config['env'])
    
    # RL config
    rl_config = RLConfig(**config['rl'])
    
    # Model config
    model_config = config['model']
    
    # Training config
    training_config = config['training']
    
    return env_config, rl_config, model_config, training_config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RL agent for text generation")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="Number of training episodes (overrides config)"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["reinforce", "ppo"],
        help="RL algorithm to use (overrides config)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["transformer", "lstm"],
        help="Model type to use (overrides config)"
    )
    parser.add_argument(
        "--reward-type",
        type=str,
        choices=["length", "coherence", "diversity"],
        help="Reward function type (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use (overrides config)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use Weights & Biases logging"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation (requires trained model)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint file for evaluation"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.episodes is not None:
        config['training']['num_episodes'] = args.episodes
    if args.algorithm is not None:
        config['rl']['algorithm'] = args.algorithm
    if args.model_type is not None:
        config['model']['model_type'] = args.model_type
    if args.reward_type is not None:
        config['env']['reward_type'] = args.reward_type
    if args.device is not None:
        config['rl']['device'] = args.device
    if args.seed is not None:
        config['env']['seed'] = args.seed
    if args.wandb:
        config['training']['use_wandb'] = True
    
    # Create configuration objects
    env_config, rl_config, model_config, training_config = create_configs(config)
    
    # Print configuration
    print("=" * 60)
    print("RL Text Generation Training Configuration")
    print("=" * 60)
    print(f"Algorithm: {rl_config.algorithm}")
    print(f"Model Type: {model_config['model_type']}")
    print(f"Reward Type: {env_config.reward_type}")
    print(f"Vocabulary Size: {env_config.vocab_size}")
    print(f"Max Length: {env_config.max_length}")
    print(f"Learning Rate: {rl_config.learning_rate}")
    print(f"Device: {rl_config.device}")
    print(f"Episodes: {training_config['num_episodes']}")
    print(f"Seed: {env_config.seed}")
    print("=" * 60)
    
    # Initialize trainer
    trainer = RLTrainer(
        env_config=env_config,
        rl_config=rl_config,
        model_config=model_config,
        log_dir=training_config['log_dir'],
        use_wandb=training_config['use_wandb'],
        wandb_project=training_config['wandb_project']
    )
    
    try:
        if args.eval_only:
            # Evaluation only mode
            if args.checkpoint:
                print(f"Loading checkpoint: {args.checkpoint}")
                trainer.load_checkpoint(args.checkpoint)
            
            print("Running evaluation...")
            eval_metrics = trainer.evaluate(num_episodes=config['evaluation']['num_eval_episodes'])
            
            print("\nEvaluation Results:")
            print("-" * 40)
            for key, value in eval_metrics.items():
                print(f"{key}: {value:.4f}")
            
            # Generate sample texts
            print("\nSample Generated Texts:")
            print("-" * 40)
            for i in range(config['evaluation']['num_sample_texts']):
                sample_text = trainer.generate_sample_text()
                print(f"Sample {i+1}: {sample_text}")
                
        else:
            # Training mode
            print("Starting training...")
            training_metrics = trainer.train(
                num_episodes=training_config['num_episodes'],
                eval_frequency=training_config['eval_frequency']
            )
            
            print("\nTraining completed!")
            print(f"Final episode reward: {training_metrics['episode_rewards'][-1]:.4f}")
            print(f"Final text quality: {training_metrics['text_qualities'][-1]:.4f}")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
