"""Training and evaluation utilities for RL text generation.

This module provides comprehensive training loops, evaluation metrics,
and logging utilities for RL-based text generation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from datetime import datetime
import wandb
from tqdm import tqdm
import random

from ..environments.nlg_env import NLGEnvironment, NLGConfig
from ..models.text_generators import PolicyNetwork
from ..algorithms.rl_agents import RLAgentFactory, RLConfig


class MetricsTracker:
    """Track and compute various metrics during training."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "policy_losses": [],
            "value_losses": [],
            "entropies": [],
            "returns": [],
            "advantages": [],
            "text_qualities": []
        }
        
    def update(self, metrics: Dict[str, float]):
        """Update metrics with new values."""
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of all metrics."""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[f"{key}_mean"] = np.mean(values)
                summary[f"{key}_std"] = np.std(values)
                summary[f"{key}_min"] = np.min(values)
                summary[f"{key}_max"] = np.max(values)
        return summary
    
    def reset(self):
        """Reset all metrics."""
        for key in self.metrics:
            self.metrics[key] = []


class TextQualityEvaluator:
    """Evaluate quality of generated text."""
    
    def __init__(self, vocab):
        """Initialize text quality evaluator.
        
        Args:
            vocab: Vocabulary object
        """
        self.vocab = vocab
        
    def evaluate_coherence(self, text: str) -> float:
        """Evaluate text coherence (simplified)."""
        words = text.split()
        if len(words) < 2:
            return 0.0
            
        # Simple coherence based on common word patterns
        coherence_score = 0.0
        for i in range(len(words) - 1):
            current_word = words[i].lower()
            next_word = words[i + 1].lower()
            
            # Check for common patterns
            if current_word in ["the", "a", "an"] and next_word not in ["the", "a", "an"]:
                coherence_score += 0.1
            elif current_word in ["is", "are", "was", "were"] and next_word not in ["the", "a", "an"]:
                coherence_score += 0.1
            elif current_word in ["and", "or", "but"]:
                coherence_score += 0.05
                
        return min(coherence_score, 1.0)
    
    def evaluate_diversity(self, text: str) -> float:
        """Evaluate vocabulary diversity."""
        words = text.split()
        if not words:
            return 0.0
            
        unique_words = len(set(words))
        total_words = len(words)
        return unique_words / total_words
    
    def evaluate_length(self, text: str) -> float:
        """Evaluate text length appropriateness."""
        words = text.split()
        length = len(words)
        
        # Optimal length range: 5-20 words
        if 5 <= length <= 20:
            return 1.0
        elif length < 5:
            return length / 5.0
        else:
            return max(0.0, 1.0 - (length - 20) / 10.0)
    
    def evaluate_overall(self, text: str) -> Dict[str, float]:
        """Evaluate overall text quality."""
        return {
            "coherence": self.evaluate_coherence(text),
            "diversity": self.evaluate_diversity(text),
            "length": self.evaluate_length(text),
            "overall": (self.evaluate_coherence(text) + 
                       self.evaluate_diversity(text) + 
                       self.evaluate_length(text)) / 3.0
        }


class RLTrainer:
    """Main trainer class for RL text generation."""
    
    def __init__(
        self,
        env_config: NLGConfig,
        rl_config: RLConfig,
        model_config: Dict[str, Any],
        log_dir: str = "logs",
        use_wandb: bool = False,
        wandb_project: str = "rl-text-generation"
    ):
        """Initialize RL trainer.
        
        Args:
            env_config: Environment configuration
            rl_config: RL algorithm configuration
            model_config: Model configuration
            log_dir: Directory for logging
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
        """
        self.env_config = env_config
        self.rl_config = rl_config
        self.model_config = model_config
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        
        # Set random seeds
        self._set_seeds(env_config.seed)
        
        # Initialize environment
        self.env = NLGEnvironment(env_config)
        
        # Initialize model
        self.policy_network = PolicyNetwork(
            vocab_size=env_config.vocab_size,
            **model_config
        )
        
        # Initialize agent
        self.agent = RLAgentFactory.create_agent(self.policy_network, rl_config)
        
        # Initialize evaluator
        self.evaluator = TextQualityEvaluator(self.env.vocab)
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker()
        
        # Initialize logging
        self._setup_logging()
        
        if use_wandb:
            wandb.init(project=wandb_project, config={
                "env_config": env_config.__dict__,
                "rl_config": rl_config.__dict__,
                "model_config": model_config
            })
    
    def _set_seeds(self, seed: Optional[int]):
        """Set random seeds for reproducibility."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
    
    def _setup_logging(self):
        """Setup logging directory and files."""
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "training.log")
        
        # Create subdirectories
        os.makedirs(os.path.join(self.log_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "generated_texts"), exist_ok=True)
    
    def train(self, num_episodes: int, eval_frequency: int = 100) -> Dict[str, List[float]]:
        """Train the RL agent.
        
        Args:
            num_episodes: Number of training episodes
            eval_frequency: Frequency of evaluation
            
        Returns:
            Training metrics
        """
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Using device: {self.agent.device}")
        print(f"Algorithm: {self.rl_config.algorithm}")
        
        training_metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "policy_losses": [],
            "value_losses": [],
            "text_qualities": []
        }
        
        for episode in tqdm(range(num_episodes), desc="Training"):
            # Run episode
            episode_metrics = self._run_episode(training=True)
            
            # Update metrics
            for key, value in episode_metrics.items():
                if key in training_metrics:
                    training_metrics[key].append(value)
            
            # Update agent
            if len(self.agent.buffer) >= self.rl_config.batch_size:
                experiences = self.agent.buffer.sample(self.rl_config.batch_size)
                update_metrics = self.agent.update(experiences)
                
                # Log update metrics
                if self.use_wandb:
                    wandb.log({
                        "episode": episode,
                        **update_metrics,
                        **episode_metrics
                    })
                
                # Print progress
                if episode % 50 == 0:
                    print(f"Episode {episode}: Reward={episode_metrics['episode_reward']:.3f}, "
                          f"Length={episode_metrics['episode_length']:.1f}, "
                          f"Quality={episode_metrics['text_quality']:.3f}")
            
            # Evaluation
            if episode % eval_frequency == 0 and episode > 0:
                eval_metrics = self.evaluate(num_episodes=10)
                print(f"Evaluation at episode {episode}: {eval_metrics}")
                
                # Save checkpoint
                self.save_checkpoint(episode)
                
                # Generate sample text
                sample_text = self.generate_sample_text()
                print(f"Sample generated text: {sample_text}")
        
        # Final evaluation
        final_eval = self.evaluate(num_episodes=50)
        print(f"Final evaluation: {final_eval}")
        
        # Save final model
        self.save_checkpoint(num_episodes, is_final=True)
        
        # Generate plots
        self.plot_training_curves(training_metrics)
        
        return training_metrics
    
    def _run_episode(self, training: bool = True) -> Dict[str, float]:
        """Run a single episode.
        
        Args:
            training: Whether in training mode
            
        Returns:
            Episode metrics
        """
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.long).unsqueeze(0).to(self.agent.device)
        
        episode_reward = 0.0
        episode_length = 0
        experiences = []
        
        done = False
        while not done:
            # Select action
            if self.rl_config.algorithm == "ppo":
                action, log_prob, value = self.agent.select_action(state, training=training)
            else:  # REINFORCE
                action, log_prob = self.agent.select_action(state, training=training)
                value = torch.tensor(0.0)  # Dummy value for REINFORCE
            
            # Take step
            next_state, reward, terminated, truncated, info = self.env.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.long).unsqueeze(0).to(self.agent.device)
            
            # Store experience
            if training:
                if self.rl_config.algorithm == "ppo":
                    self.agent.store_experience(state, action, reward, next_state, 
                                             terminated or truncated, log_prob, value)
                else:  # REINFORCE
                    self.agent.store_experience(state, action, reward, next_state, 
                                             terminated or truncated, log_prob)
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            done = terminated or truncated
        
        # Evaluate text quality
        generated_text = info["text"]
        quality_metrics = self.evaluator.evaluate_overall(generated_text)
        
        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "text_quality": quality_metrics["overall"],
            "coherence": quality_metrics["coherence"],
            "diversity": quality_metrics["diversity"],
            "length_score": quality_metrics["length"]
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the trained agent.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        self.policy_network.eval()
        
        eval_metrics = {
            "rewards": [],
            "lengths": [],
            "qualities": [],
            "coherences": [],
            "diversities": [],
            "length_scores": []
        }
        
        with torch.no_grad():
            for _ in range(num_episodes):
                episode_metrics = self._run_episode(training=False)
                
                eval_metrics["rewards"].append(episode_metrics["episode_reward"])
                eval_metrics["lengths"].append(episode_metrics["episode_length"])
                eval_metrics["qualities"].append(episode_metrics["text_quality"])
                eval_metrics["coherences"].append(episode_metrics["coherence"])
                eval_metrics["diversities"].append(episode_metrics["diversity"])
                eval_metrics["length_scores"].append(episode_metrics["length_score"])
        
        # Compute summary statistics
        summary = {}
        for key, values in eval_metrics.items():
            summary[f"eval_{key}_mean"] = np.mean(values)
            summary[f"eval_{key}_std"] = np.std(values)
            summary[f"eval_{key}_ci"] = 1.96 * np.std(values) / np.sqrt(len(values))
        
        self.policy_network.train()
        return summary
    
    def generate_sample_text(self, prompt: Optional[str] = None) -> str:
        """Generate sample text using the trained model.
        
        Args:
            prompt: Optional prompt text
            
        Returns:
            Generated text
        """
        self.policy_network.eval()
        
        if prompt is None:
            # Start with a random word
            prompt_idx = random.randint(0, self.env.vocab.vocab_size - 1)
        else:
            prompt_indices = self.env.vocab.encode(prompt)
            if not prompt_indices:
                prompt_idx = random.randint(0, self.env.vocab.vocab_size - 1)
            else:
                prompt_idx = prompt_indices[0]
        
        prompt_tensor = torch.tensor([[prompt_idx]], dtype=torch.long).to(self.agent.device)
        
        with torch.no_grad():
            generated = self.policy_network.generate(
                prompt=prompt_tensor,
                max_length=self.env_config.max_length,
                temperature=0.8,
                do_sample=True
            )
        
        generated_text = self.env.vocab.decode(generated[0].cpu().tolist())
        self.policy_network.train()
        
        return generated_text
    
    def save_checkpoint(self, episode: int, is_final: bool = False):
        """Save model checkpoint.
        
        Args:
            episode: Current episode number
            is_final: Whether this is the final checkpoint
        """
        checkpoint = {
            "episode": episode,
            "policy_network_state_dict": self.policy_network.state_dict(),
            "env_config": self.env_config.__dict__,
            "rl_config": self.rl_config.__dict__,
            "model_config": self.model_config
        }
        
        if self.rl_config.algorithm == "ppo":
            checkpoint["value_network_state_dict"] = self.agent.value_network.state_dict()
        
        filename = f"checkpoint_episode_{episode}.pt" if not is_final else "final_model.pt"
        filepath = os.path.join(self.log_dir, "checkpoints", filename)
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.agent.device)
        
        self.policy_network.load_state_dict(checkpoint["policy_network_state_dict"])
        
        if self.rl_config.algorithm == "ppo" and "value_network_state_dict" in checkpoint:
            self.agent.value_network.load_state_dict(checkpoint["value_network_state_dict"])
        
        print(f"Checkpoint loaded: {filepath}")
        return checkpoint["episode"]
    
    def plot_training_curves(self, metrics: Dict[str, List[float]]):
        """Plot training curves.
        
        Args:
            metrics: Training metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(metrics["episode_rewards"])
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True)
        
        # Episode lengths
        axes[0, 1].plot(metrics["episode_lengths"])
        axes[0, 1].set_title("Episode Lengths")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Length")
        axes[0, 1].grid(True)
        
        # Text qualities
        axes[1, 0].plot(metrics["text_qualities"])
        axes[1, 0].set_title("Text Quality")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Quality Score")
        axes[1, 0].grid(True)
        
        # Policy losses
        if metrics["policy_losses"]:
            axes[1, 1].plot(metrics["policy_losses"])
            axes[1, 1].set_title("Policy Loss")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.log_dir, "plots", "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"Training curves saved: {plot_path}")
    
    def close(self):
        """Close the trainer and cleanup resources."""
        if self.use_wandb:
            wandb.finish()
        self.env.close()
