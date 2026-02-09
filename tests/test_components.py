"""Unit tests for the RL text generation project."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.environments.nlg_env import NLGEnvironment, NLGConfig, Vocabulary
from src.models.text_generators import PolicyNetwork, TransformerTextGenerator, LSTMTextGenerator
from src.algorithms.rl_agents import REINFORCEAgent, PPOAgent, RLConfig
from src.utils.helpers import set_seed, get_device, compute_confidence_interval


class TestVocabulary:
    """Test vocabulary functionality."""
    
    def test_vocabulary_initialization(self):
        """Test vocabulary initialization."""
        vocab = Vocabulary(vocab_size=100)
        assert vocab.vocab_size >= 100
        assert len(vocab.word_to_idx) >= 100
        assert len(vocab.idx_to_word) >= 100
    
    def test_encode_decode(self):
        """Test text encoding and decoding."""
        vocab = Vocabulary(vocab_size=100)
        
        # Test encoding
        text = "the cat is happy"
        indices = vocab.encode(text)
        assert isinstance(indices, list)
        assert len(indices) > 0
        
        # Test decoding
        decoded = vocab.decode(indices)
        assert isinstance(decoded, str)
    
    def test_special_tokens(self):
        """Test special token handling."""
        vocab = Vocabulary(vocab_size=100)
        
        # Check special tokens exist
        assert vocab.pad_token in vocab.word_to_idx
        assert vocab.start_token in vocab.word_to_idx
        assert vocab.end_token in vocab.word_to_idx
        assert vocab.unk_token in vocab.word_to_idx


class TestNLGEnvironment:
    """Test NLG environment functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = NLGConfig(
            vocab_size=100,
            max_length=10,
            min_length=2,
            reward_type="length",
            seed=42
        )
        self.env = NLGEnvironment(self.config)
    
    def test_environment_initialization(self):
        """Test environment initialization."""
        assert self.env.config.vocab_size == 100
        assert self.env.config.max_length == 10
        assert self.env.action_space.n == 100
        assert self.env.observation_space.shape == (10,)
    
    def test_reset(self):
        """Test environment reset."""
        obs, info = self.env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (10,)
        assert isinstance(info, dict)
        assert "text" in info
        assert "step_count" in info
        assert "done" in info
    
    def test_step(self):
        """Test environment step."""
        obs, _ = self.env.reset()
        
        # Take a step
        action = 0  # Valid action
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        
        assert isinstance(next_obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_reward_functions(self):
        """Test different reward functions."""
        # Test length reward
        env_length = NLGEnvironment(NLGConfig(reward_type="length"))
        text = [1, 2, 3, 4, 5]
        reward = env_length._length_reward(text)
        assert isinstance(reward, float)
        assert 0 <= reward <= 1
        
        # Test coherence reward
        env_coherence = NLGEnvironment(NLGConfig(reward_type="coherence"))
        reward = env_coherence._coherence_reward(text)
        assert isinstance(reward, float)
        assert 0 <= reward <= 1
        
        # Test diversity reward
        env_diversity = NLGEnvironment(NLGConfig(reward_type="diversity"))
        reward = env_diversity._diversity_reward(text)
        assert isinstance(reward, float)
        assert 0 <= reward <= 1


class TestTextGenerators:
    """Test text generator models."""
    
    def test_transformer_generator(self):
        """Test transformer text generator."""
        model = TransformerTextGenerator(
            vocab_size=100,
            d_model=64,
            nhead=4,
            num_layers=2,
            max_length=20
        )
        
        # Test forward pass
        x = torch.randint(0, 100, (2, 10))
        logits = model(x)
        
        assert logits.shape == (2, 10, 100)
    
    def test_lstm_generator(self):
        """Test LSTM text generator."""
        model = LSTMTextGenerator(
            vocab_size=100,
            hidden_size=64,
            num_layers=2
        )
        
        # Test forward pass
        x = torch.randint(0, 100, (2, 10))
        logits, hidden = model(x)
        
        assert logits.shape == (2, 10, 100)
        assert isinstance(hidden, tuple)
        assert len(hidden) == 2
    
    def test_policy_network(self):
        """Test policy network wrapper."""
        model = PolicyNetwork(
            vocab_size=100,
            hidden_size=64,
            model_type="lstm"
        )
        
        # Test action logits
        state = torch.randint(0, 100, (2, 10))
        logits = model.get_action_logits(state)
        
        assert logits.shape == (2, 100)
        
        # Test action probabilities
        probs = model.get_action_probs(state)
        assert probs.shape == (2, 100)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2))
        
        # Test action sampling
        action = model.sample_action(state)
        assert action.shape == (2,)


class TestRLAgents:
    """Test RL agents."""
    
    def setup_method(self):
        """Setup test agents."""
        self.policy_network = PolicyNetwork(
            vocab_size=100,
            hidden_size=64,
            model_type="lstm"
        )
        self.config = RLConfig(
            algorithm="reinforce",
            learning_rate=1e-3,
            batch_size=32
        )
    
    def test_reinforce_agent(self):
        """Test REINFORCE agent."""
        agent = REINFORCEAgent(self.policy_network, self.config)
        
        # Test action selection
        state = torch.randint(0, 100, (1, 10))
        action, log_prob = agent.select_action(state)
        
        assert isinstance(action, torch.Tensor)
        assert isinstance(log_prob, torch.Tensor)
        assert action.shape == (1,)
        assert log_prob.shape == (1,)
    
    def test_ppo_agent(self):
        """Test PPO agent."""
        config = RLConfig(algorithm="ppo", learning_rate=1e-3, batch_size=32)
        agent = PPOAgent(self.policy_network, config)
        
        # Test action selection
        state = torch.randint(0, 100, (1, 10))
        action, log_prob, value = agent.select_action(state)
        
        assert isinstance(action, torch.Tensor)
        assert isinstance(log_prob, torch.Tensor)
        assert isinstance(value, torch.Tensor)
        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert value.shape == (1,)


class TestUtilities:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test numpy
        np_val1 = np.random.random()
        set_seed(42)
        np_val2 = np.random.random()
        assert np_val1 == np_val2
        
        # Test torch
        torch_val1 = torch.randn(1)
        set_seed(42)
        torch_val2 = torch.randn(1)
        assert torch.equal(torch_val1, torch_val2)
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("cpu")
        assert device.type == "cpu"
        
        device = get_device("auto")
        assert device.type in ["cpu", "cuda", "mps"]
    
    def test_confidence_interval(self):
        """Test confidence interval computation."""
        data = [1, 2, 3, 4, 5]
        mean, margin = compute_confidence_interval(data)
        
        assert isinstance(mean, float)
        assert isinstance(margin, float)
        assert mean == 3.0  # Expected mean
        assert margin > 0  # Should have some margin


class TestIntegration:
    """Integration tests."""
    
    def test_full_training_step(self):
        """Test a complete training step."""
        # Setup
        config = NLGConfig(vocab_size=50, max_length=5, seed=42)
        env = NLGEnvironment(config)
        
        policy_network = PolicyNetwork(
            vocab_size=50,
            hidden_size=32,
            model_type="lstm"
        )
        
        rl_config = RLConfig(algorithm="reinforce", learning_rate=1e-3, batch_size=16)
        agent = REINFORCEAgent(policy_network, rl_config)
        
        # Run episode
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.long).unsqueeze(0)
        
        episode_reward = 0
        done = False
        
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.long).unsqueeze(0)
            
            agent.store_experience(state, action, reward, next_state, 
                                terminated or truncated, log_prob)
            
            state = next_state
            episode_reward += reward
            done = terminated or truncated
        
        # Test that we collected some experiences
        assert len(agent.buffer) > 0
        
        # Test update
        if len(agent.buffer) >= rl_config.batch_size:
            experiences = agent.buffer.sample(rl_config.batch_size)
            metrics = agent.update(experiences)
            
            assert isinstance(metrics, dict)
            assert "loss" in metrics


if __name__ == "__main__":
    pytest.main([__file__])
