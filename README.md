# RL for Natural Language Generation

A comprehensive Reinforcement Learning project for Natural Language Generation (NLG) using modern RL algorithms and neural network architectures.

## ⚠️ IMPORTANT DISCLAIMER

**This project is for RESEARCH and EDUCATIONAL purposes only. It is NOT suitable for production use or real-world text generation tasks. The models are trained on synthetic data and should not be used for any production applications.**

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/RL-for-Natural-Language-Generation.git
cd RL-for-Natural-Language-Generation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the interactive demo:
```bash
streamlit run demo/app.py
```

### Basic Training

Train a simple REINFORCE model:
```bash
python scripts/train.py --config configs/simple.yaml --episodes 200
```

Train a PPO model with transformer:
```bash
python scripts/train.py --config configs/default.yaml --algorithm ppo --model-type transformer --episodes 500
```

## 📁 Project Structure

```
├── src/                          # Source code
│   ├── algorithms/               # RL algorithms (REINFORCE, PPO)
│   ├── environments/             # NLG environment and vocabulary
│   ├── models/                   # Neural network models (Transformer, LSTM)
│   ├── training/                 # Training utilities and metrics
│   └── utils/                    # Utility functions
├── configs/                      # Configuration files
│   ├── default.yaml             # Default configuration
│   └── simple.yaml              # Simple configuration
├── scripts/                      # Training and evaluation scripts
│   └── train.py                 # Main training script
├── demo/                        # Interactive Streamlit demo
│   └── app.py                   # Demo application
├── tests/                       # Unit tests
├── notebooks/                   # Jupyter notebooks for analysis
├── assets/                      # Generated plots and videos
├── data/                        # Data directory
├── logs/                        # Training logs and checkpoints
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

## Algorithms and Models

### Supported RL Algorithms

- **REINFORCE**: Policy gradient method with baseline
- **PPO**: Proximal Policy Optimization with actor-critic architecture

### Supported Model Architectures

- **Transformer**: Multi-head attention with positional encoding
- **LSTM**: Long Short-Term Memory networks

### Reward Functions

- **Length**: Rewards longer generated text
- **Coherence**: Rewards coherent word transitions
- **Diversity**: Rewards vocabulary diversity

## Features

### Core Features
- Modern RL algorithms (REINFORCE, PPO)
- Multiple neural network architectures
- Comprehensive evaluation metrics
- Interactive Streamlit demo
- Configurable training parameters
- Real-time training visualization

### Evaluation Metrics
- Episode rewards and lengths
- Text quality scores (coherence, diversity, length)
- Policy and value losses
- Training stability metrics

### Visualization
- Interactive training curves
- Real-time metrics tracking
- Generated text samples
- Performance comparisons

## Usage Examples

### Training with Different Configurations

```bash
# Simple LSTM with REINFORCE
python scripts/train.py --config configs/simple.yaml

# Transformer with PPO
python scripts/train.py --algorithm ppo --model-type transformer --reward-type coherence

# Custom parameters
python scripts/train.py --episodes 1000 --learning-rate 1e-3 --device cuda
```

### Evaluation Only

```bash
# Evaluate a trained model
python scripts/train.py --eval-only --checkpoint logs/checkpoints/final_model.pt
```

### Interactive Demo

```bash
# Launch the Streamlit demo
streamlit run demo/app.py
```

## 🔧 Configuration

### Environment Configuration

```yaml
env:
  vocab_size: 1000          # Vocabulary size
  max_length: 50            # Maximum sequence length
  min_length: 5             # Minimum sequence length
  reward_type: "length"     # Reward function type
  temperature: 1.0          # Sampling temperature
  seed: 42                  # Random seed
```

### RL Algorithm Configuration

```yaml
rl:
  algorithm: "ppo"          # Algorithm type
  learning_rate: 3e-4       # Learning rate
  gamma: 0.99               # Discount factor
  batch_size: 64            # Training batch size
  device: "auto"            # Device selection
```

### Model Configuration

```yaml
model:
  model_type: "transformer" # Model architecture
  hidden_size: 256          # Hidden state size
  num_layers: 6            # Number of layers
  dropout: 0.1             # Dropout rate
```

## Expected Results

### Training Performance

| Algorithm | Model | Episodes | Final Reward | Text Quality |
|-----------|-------|----------|--------------|--------------|
| REINFORCE | LSTM  | 200      | ~0.3-0.5     | ~0.4-0.6     |
| PPO       | LSTM  | 200      | ~0.4-0.6     | ~0.5-0.7     |
| REINFORCE | Transformer | 500  | ~0.5-0.7     | ~0.6-0.8     |
| PPO       | Transformer | 500  | ~0.6-0.8     | ~0.7-0.9     |

### Sample Generated Text

```
Sample 1: the cat is happy and plays with the ball
Sample 2: a beautiful house stands near the tall tree
Sample 3: the dog runs fast through the green park
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Run specific tests:
```bash
pytest tests/test_environment.py
pytest tests/test_algorithms.py
```

## Educational Value

This project demonstrates:

1. **RL Fundamentals**: Policy gradients, value functions, advantage estimation
2. **Text Generation**: Autoregressive generation, vocabulary management
3. **Neural Architectures**: Transformers vs LSTMs for sequence modeling
4. **Reward Design**: Different reward functions and their impact
5. **Training Dynamics**: Learning curves, stability, convergence
6. **Evaluation**: Multiple metrics for text quality assessment

## Research Applications

- Understanding RL for discrete action spaces
- Comparing policy gradient methods
- Analyzing reward function design
- Studying exploration vs exploitation in text generation
- Investigating sample efficiency in RL

## Development

### Code Quality

The project uses:
- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **Type hints**: Full type annotations
- **Docstrings**: Comprehensive documentation

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Requirements

### System Requirements
- Python 3.10+
- PyTorch 2.0+
- CUDA/MPS support (optional)

### Dependencies
- torch>=2.0.0
- numpy>=1.24.0
- gymnasium>=0.29.0
- streamlit>=1.25.0
- matplotlib>=3.7.0
- plotly>=5.15.0
- omegaconf>=2.3.0

## Safety and Limitations

### Limitations
- **Synthetic Data**: Trained on artificial vocabulary, not real text
- **Simple Rewards**: Basic reward functions for demonstration
- **Limited Vocabulary**: Only 1000 common English words
- **No Real-world Applicability**: Not suitable for production use

### Safety Considerations
- Models may generate nonsensical text
- No content filtering or safety mechanisms
- Not trained on real-world text data
- Results are for educational purposes only

## References

1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction.
2. Schulman, J., et al. (2017). Proximal policy optimization algorithms.
3. Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning.
4. Vaswani, A., et al. (2017). Attention is all you need.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for RL research and algorithms
- PyTorch team for the deep learning framework
- Streamlit team for the interactive demo framework
- The RL research community for foundational work

---

**Remember**: This is a research/educational project. Do not use for production applications.
# RL-for-Natural-Language-Generation
