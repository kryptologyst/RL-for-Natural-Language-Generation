#!/usr/bin/env python3
"""Simple example script demonstrating RL text generation.

This script provides a minimal example of training an RL agent
for text generation with default parameters.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.environments.nlg_env import NLGConfig
from src.algorithms.rl_agents import RLConfig
from src.training.trainer import RLTrainer


def main():
    """Run a simple training example."""
    print("RL Text Generation - Simple Example")
    print("=" * 50)
    
    # Configuration
    env_config = NLGConfig(
        vocab_size=500,
        max_length=15,
        min_length=3,
        reward_type="length",
        seed=42
    )
    
    rl_config = RLConfig(
        algorithm="reinforce",
        learning_rate=1e-3,
        gamma=0.99,
        batch_size=32,
        device="auto"
    )
    
    model_config = {
        "model_type": "lstm",
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.1
    }
    
    # Initialize trainer
    trainer = RLTrainer(
        env_config=env_config,
        rl_config=rl_config,
        model_config=model_config,
        log_dir="example_logs",
        use_wandb=False
    )
    
    try:
        # Train for a few episodes
        print("Starting training...")
        training_metrics = trainer.train(num_episodes=100, eval_frequency=25)
        
        # Evaluate
        print("\nEvaluating...")
        eval_metrics = trainer.evaluate(num_episodes=20)
        
        # Generate samples
        print("\nGenerated samples:")
        for i in range(3):
            sample = trainer.generate_sample_text()
            print(f"  {i+1}. {sample}")
        
        # Print results
        print(f"\nFinal reward: {training_metrics['episode_rewards'][-1]:.4f}")
        print(f"Final quality: {training_metrics['text_qualities'][-1]:.4f}")
        print(f"Average length: {sum(training_metrics['episode_lengths'])/len(training_metrics['episode_lengths']):.2f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        trainer.close()
    
    print("\nExample completed!")


if __name__ == "__main__":
    main()
