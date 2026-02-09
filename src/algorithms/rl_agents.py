"""Reinforcement Learning algorithms for Natural Language Generation.

This module implements various RL algorithms specifically designed for text generation
tasks, including REINFORCE, PPO, and baseline methods.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math
from collections import deque


@dataclass
class RLConfig:
    """Configuration for RL algorithms."""
    
    # Algorithm parameters
    algorithm: str = "ppo"  # "reinforce", "ppo", "a2c"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lambda_gae: float = 0.95
    
    # PPO specific
    ppo_clip_ratio: float = 0.2
    ppo_epochs: int = 4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training parameters
    batch_size: int = 64
    buffer_size: int = 10000
    update_frequency: int = 10
    
    # Exploration
    temperature: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"


class ValueNetwork(nn.Module):
    """Value network for actor-critic methods."""
    
    def __init__(self, vocab_size: int, hidden_size: int = 256, device: Optional[torch.device] = None):
        """Initialize value network.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Hidden state size
            device: Device to run on
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.device = device or torch.device("cpu")
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through value network.
        
        Args:
            x: Input sequence [batch_size, seq_len]
            
        Returns:
            Value estimates [batch_size, seq_len, 1]
        """
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(embedded)
        
        # Value prediction
        values = self.value_head(lstm_out)
        
        return values


class ExperienceBuffer:
    """Experience buffer for storing trajectories."""
    
    def __init__(self, buffer_size: int):
        """Initialize experience buffer.
        
        Args:
            buffer_size: Maximum buffer size
        """
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        
    def add(self, experience: Dict[str, Any]):
        """Add experience to buffer."""
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample batch of experiences."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return np.random.choice(list(self.buffer), batch_size, replace=False).tolist()
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
    
    def __len__(self) -> int:
        return len(self.buffer)


class REINFORCEAgent:
    """REINFORCE agent for text generation."""
    
    def __init__(self, policy_network: nn.Module, config: RLConfig):
        """Initialize REINFORCE agent.
        
        Args:
            policy_network: Policy network
            config: RL configuration
        """
        self.policy_network = policy_network
        self.config = config
        
        # Set device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                     "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
            
        self.policy_network.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=config.learning_rate
        )
        
        # Experience buffer
        self.buffer = ExperienceBuffer(config.buffer_size)
        
        # Training state
        self.episode_rewards = []
        self.episode_lengths = []
        
    def select_action(self, state: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select action using policy network.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Action and log probability
        """
        with torch.no_grad() if not training else torch.enable_grad():
            logits = self.policy_network.get_action_logits(state)
            
            if training:
                # Sample action
                probs = torch.softmax(logits / self.config.temperature, dim=-1)
                action = torch.multinomial(probs, 1).squeeze(-1)
                log_prob = torch.log(probs.gather(1, action.unsqueeze(-1))).squeeze(-1)
            else:
                # Greedy action
                action = torch.argmax(logits, dim=-1)
                log_prob = torch.log(torch.softmax(logits, dim=-1).gather(1, action.unsqueeze(-1))).squeeze(-1)
            
            return action, log_prob
    
    def store_experience(self, state: torch.Tensor, action: torch.Tensor, 
                        reward: float, next_state: torch.Tensor, done: bool, log_prob: torch.Tensor):
        """Store experience in buffer."""
        experience = {
            "state": state.cpu(),
            "action": action.cpu(),
            "reward": reward,
            "next_state": next_state.cpu(),
            "done": done,
            "log_prob": log_prob.cpu()
        }
        self.buffer.add(experience)
    
    def compute_returns(self, rewards: List[float]) -> List[float]:
        """Compute discounted returns."""
        returns = []
        discounted_reward = 0
        
        for reward in reversed(rewards):
            discounted_reward = reward + self.config.gamma * discounted_reward
            returns.insert(0, discounted_reward)
            
        return returns
    
    def update(self, experiences: List[Dict[str, Any]]) -> Dict[str, float]:
        """Update policy using REINFORCE.
        
        Args:
            experiences: List of experiences
            
        Returns:
            Training metrics
        """
        if not experiences:
            return {"loss": 0.0, "policy_loss": 0.0}
        
        # Extract data
        states = torch.stack([exp["state"] for exp in experiences]).to(self.device)
        actions = torch.stack([exp["action"] for exp in experiences]).to(self.device)
        rewards = [exp["reward"] for exp in experiences]
        log_probs = torch.stack([exp["log_prob"] for exp in experiences]).to(self.device)
        
        # Compute returns
        returns = self.compute_returns(rewards)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute policy loss
        current_log_probs = self.policy_network.get_log_prob(states, actions)
        policy_loss = -(current_log_probs * returns).mean()
        
        # Add entropy regularization
        probs = self.policy_network.get_action_probs(states)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        total_loss = policy_loss - self.config.entropy_coef * entropy
        
        # Update policy
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        return {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "returns_mean": returns.mean().item(),
            "returns_std": returns.std().item()
        }


class PPOAgent:
    """PPO agent for text generation."""
    
    def __init__(self, policy_network: nn.Module, config: RLConfig):
        """Initialize PPO agent.
        
        Args:
            policy_network: Policy network
            config: RL configuration
        """
        self.policy_network = policy_network
        self.config = config
        
        # Set device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                     "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
            
        self.policy_network.to(self.device)
        
        # Value network
        self.value_network = ValueNetwork(
            vocab_size=policy_network.vocab_size,
            hidden_size=policy_network.hidden_size,
            device=self.device
        )
        
        # Optimizers
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=config.learning_rate
        )
        self.value_optimizer = optim.Adam(
            self.value_network.parameters(),
            lr=config.learning_rate
        )
        
        # Experience buffer
        self.buffer = ExperienceBuffer(config.buffer_size)
        
        # Training state
        self.episode_rewards = []
        self.episode_lengths = []
        
    def select_action(self, state: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select action using policy network.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Action, log probability, and value estimate
        """
        with torch.no_grad() if not training else torch.enable_grad():
            logits = self.policy_network.get_action_logits(state)
            values = self.value_network(state)
            
            if training:
                # Sample action
                probs = torch.softmax(logits / self.config.temperature, dim=-1)
                action = torch.multinomial(probs, 1).squeeze(-1)
                log_prob = torch.log(probs.gather(1, action.unsqueeze(-1))).squeeze(-1)
            else:
                # Greedy action
                action = torch.argmax(logits, dim=-1)
                log_prob = torch.log(torch.softmax(logits, dim=-1).gather(1, action.unsqueeze(-1))).squeeze(-1)
            
            return action, log_prob, values[:, -1, 0]
    
    def store_experience(self, state: torch.Tensor, action: torch.Tensor, 
                        reward: float, next_state: torch.Tensor, done: bool, 
                        log_prob: torch.Tensor, value: torch.Tensor):
        """Store experience in buffer."""
        experience = {
            "state": state.cpu(),
            "action": action.cpu(),
            "reward": reward,
            "next_state": next_state.cpu(),
            "done": done,
            "log_prob": log_prob.cpu(),
            "value": value.cpu()
        }
        self.buffer.add(experience)
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                   next_value: float = 0.0) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        returns = []
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - 0  # Assuming episode continues
                next_value = next_value
            else:
                next_non_terminal = 1.0 - 0  # Assuming episode continues
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.config.gamma * self.config.lambda_gae * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            
        return advantages, returns
    
    def update(self, experiences: List[Dict[str, Any]]) -> Dict[str, float]:
        """Update policy using PPO.
        
        Args:
            experiences: List of experiences
            
        Returns:
            Training metrics
        """
        if not experiences:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}
        
        # Extract data
        states = torch.stack([exp["state"] for exp in experiences]).to(self.device)
        actions = torch.stack([exp["action"] for exp in experiences]).to(self.device)
        rewards = [exp["reward"] for exp in experiences]
        old_log_probs = torch.stack([exp["log_prob"] for exp in experiences]).to(self.device)
        old_values = torch.stack([exp["value"] for exp in experiences]).to(self.device)
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, old_values.tolist())
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO updates
        policy_losses = []
        value_losses = []
        
        for _ in range(self.config.ppo_epochs):
            # Policy loss
            current_log_probs = self.policy_network.get_log_prob(states, actions)
            ratio = torch.exp(current_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.config.ppo_clip_ratio, 
                               1.0 + self.config.ppo_clip_ratio) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            current_values = self.value_network(states)[:, -1, 0]
            value_loss = nn.MSELoss()(current_values, returns)
            
            # Entropy bonus
            probs = self.policy_network.get_action_probs(states)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            
            # Total loss
            total_loss = policy_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy
            
            # Update networks
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.config.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.config.max_grad_norm)
            
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
        
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "total_loss": np.mean(policy_losses) + self.config.value_loss_coef * np.mean(value_losses),
            "entropy": entropy.item(),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item()
        }


class RLAgentFactory:
    """Factory for creating RL agents."""
    
    @staticmethod
    def create_agent(policy_network: nn.Module, config: RLConfig):
        """Create RL agent based on configuration.
        
        Args:
            policy_network: Policy network
            config: RL configuration
            
        Returns:
            RL agent instance
        """
        if config.algorithm == "reinforce":
            return REINFORCEAgent(policy_network, config)
        elif config.algorithm == "ppo":
            return PPOAgent(policy_network, config)
        else:
            raise ValueError(f"Unsupported algorithm: {config.algorithm}")
