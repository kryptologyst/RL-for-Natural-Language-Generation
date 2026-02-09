"""Neural network models for Natural Language Generation with RL.

This module implements various neural network architectures for text generation
in reinforcement learning settings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_length: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_length: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to input."""
        return x + self.pe[:x.size(0), :]


class TransformerTextGenerator(nn.Module):
    """Transformer-based text generator for RL."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_length: int = 100,
        device: Optional[torch.device] = None
    ):
        """Initialize transformer text generator.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            max_length: Maximum sequence length
            device: Device to run on
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_length = max_length
        self.device = device or torch.device("cpu")
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the transformer.
        
        Args:
            x: Input token indices [batch_size, seq_len]
            mask: Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Logits for next token prediction [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape
        
        # Token embeddings
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Create causal mask for autoregressive generation
        if mask is None:
            mask = self._generate_square_subsequent_mask(seq_len).to(self.device)
        
        # Transformer forward pass
        x = self.transformer(x, mask=mask)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    
    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """Generate text autoregressively.
        
        Args:
            prompt: Initial prompt [batch_size, prompt_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            Generated sequence [batch_size, max_length]
        """
        self.eval()
        batch_size = prompt.shape[0]
        device = prompt.device
        
        generated = prompt.clone()
        
        with torch.no_grad():
            for _ in range(max_length - prompt.shape[1]):
                # Get logits for next token
                logits = self.forward(generated)[:, -1, :]  # [batch_size, vocab_size]
                
                if do_sample:
                    # Apply temperature
                    logits = logits / temperature
                    
                    # Apply top-k filtering
                    if top_k is not None:
                        top_k = min(top_k, logits.size(-1))
                        topk_logits, topk_indices = torch.topk(logits, top_k)
                        logits = torch.full_like(logits, float('-inf'))
                        logits.scatter_(-1, topk_indices, topk_logits)
                    
                    # Apply top-p filtering
                    if top_p is not None:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = float('-inf')
                    
                    # Sample from the filtered distribution
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append next token
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated


class LSTMTextGenerator(nn.Module):
    """LSTM-based text generator for RL."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        device: Optional[torch.device] = None
    ):
        """Initialize LSTM text generator.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            device: Device to run on
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device or torch.device("cpu")
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the LSTM.
        
        Args:
            x: Input token indices [batch_size, seq_len]
            hidden: Previous hidden state
            
        Returns:
            Logits for next token prediction [batch_size, seq_len, vocab_size]
            Updated hidden state
        """
        # Embedding
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Output projection
        logits = self.output_projection(lstm_out)
        
        return logits, hidden
    
    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """Generate text autoregressively.
        
        Args:
            prompt: Initial prompt [batch_size, prompt_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            Generated sequence [batch_size, max_length]
        """
        self.eval()
        batch_size = prompt.shape[0]
        device = prompt.device
        
        generated = prompt.clone()
        hidden = None
        
        with torch.no_grad():
            for _ in range(max_length - prompt.shape[1]):
                # Get logits for next token
                logits, hidden = self.forward(generated[:, -1:], hidden)
                logits = logits[:, -1, :]  # [batch_size, vocab_size]
                
                if do_sample:
                    # Apply temperature
                    logits = logits / temperature
                    
                    # Apply top-k filtering
                    if top_k is not None:
                        top_k = min(top_k, logits.size(-1))
                        topk_logits, topk_indices = torch.topk(logits, top_k)
                        logits = torch.full_like(logits, float('-inf'))
                        logits.scatter_(-1, topk_indices, topk_logits)
                    
                    # Apply top-p filtering
                    if top_p is not None:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = float('-inf')
                    
                    # Sample from the filtered distribution
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append next token
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated


class PolicyNetwork(nn.Module):
    """Policy network for RL text generation."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        model_type: str = "transformer",
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """Initialize policy network.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Hidden state size
            model_type: Type of model ("transformer" or "lstm")
            device: Device to run on
            **kwargs: Additional arguments for model initialization
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.device = device or torch.device("cpu")
        
        if model_type == "transformer":
            self.model = TransformerTextGenerator(
                vocab_size=vocab_size,
                d_model=hidden_size,
                device=device,
                **kwargs
            )
        elif model_type == "lstm":
            self.model = LSTMTextGenerator(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                device=device,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.model_type = model_type
        
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the policy network."""
        return self.model(x, **kwargs)
    
    def get_action_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Get logits for action selection.
        
        Args:
            state: Current state [batch_size, seq_len]
            
        Returns:
            Action logits [batch_size, vocab_size]
        """
        if self.model_type == "transformer":
            logits = self.model(state)
            return logits[:, -1, :]  # Last token logits
        else:  # LSTM
            logits, _ = self.model(state)
            return logits[:, -1, :]  # Last token logits
    
    def get_action_probs(self, state: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Get action probabilities.
        
        Args:
            state: Current state [batch_size, seq_len]
            temperature: Temperature for softmax
            
        Returns:
            Action probabilities [batch_size, vocab_size]
        """
        logits = self.get_action_logits(state)
        return F.softmax(logits / temperature, dim=-1)
    
    def sample_action(self, state: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Sample action from policy.
        
        Args:
            state: Current state [batch_size, seq_len]
            temperature: Temperature for sampling
            
        Returns:
            Sampled action [batch_size]
        """
        probs = self.get_action_probs(state, temperature)
        return torch.multinomial(probs, 1).squeeze(-1)
    
    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get log probability of action given state.
        
        Args:
            state: Current state [batch_size, seq_len]
            action: Action taken [batch_size]
            
        Returns:
            Log probability [batch_size]
        """
        logits = self.get_action_logits(state)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
    
    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """Generate text using the policy network."""
        return self.model.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample
        )
