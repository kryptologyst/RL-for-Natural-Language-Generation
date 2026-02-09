"""Interactive Streamlit demo for RL-based Natural Language Generation.

This demo provides a user-friendly interface for:
- Training RL agents on text generation
- Generating text with trained models
- Comparing different algorithms and configurations
- Visualizing training progress and metrics
"""

import streamlit as st
import torch
import numpy as np
import yaml
import os
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.environments.nlg_env import NLGEnvironment, NLGConfig
from src.models.text_generators import PolicyNetwork
from src.algorithms.rl_agents import RLConfig, RLAgentFactory
from src.training.trainer import RLTrainer


# Page configuration
st.set_page_config(
    page_title="RL Text Generation Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_vocabulary():
    """Load vocabulary for text generation."""
    from src.environments.nlg_env import Vocabulary
    return Vocabulary(vocab_size=1000)


@st.cache_data
def load_config(config_name: str):
    """Load configuration file."""
    config_path = Path(__file__).parent.parent / "configs" / f"{config_name}.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_session_state():
    """Initialize session state variables."""
    if 'training_history' not in st.session_state:
        st.session_state.training_history = []
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None
    if 'current_config' not in st.session_state:
        st.session_state.current_config = None
    if 'generated_texts' not in st.session_state:
        st.session_state.generated_texts = []


def create_model(config: dict):
    """Create a new model based on configuration."""
    model_config = config['model']
    env_config = config['env']
    
    policy_network = PolicyNetwork(
        vocab_size=env_config['vocab_size'],
        **model_config
    )
    
    return policy_network


def train_model(config: dict, num_episodes: int, progress_bar):
    """Train the model and return training metrics."""
    # Create configuration objects
    env_config = NLGConfig(**config['env'])
    rl_config = RLConfig(**config['rl'])
    model_config = config['model']
    training_config = config['training']
    
    # Initialize trainer
    trainer = RLTrainer(
        env_config=env_config,
        rl_config=rl_config,
        model_config=model_config,
        log_dir="demo_logs",
        use_wandb=False
    )
    
    # Training metrics storage
    training_metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "text_qualities": [],
        "policy_losses": []
    }
    
    # Training loop
    for episode in range(num_episodes):
        # Run episode
        episode_metrics = trainer._run_episode(training=True)
        
        # Store metrics
        training_metrics["episode_rewards"].append(episode_metrics["episode_reward"])
        training_metrics["episode_lengths"].append(episode_metrics["episode_length"])
        training_metrics["text_qualities"].append(episode_metrics["text_quality"])
        
        # Update agent
        if len(trainer.agent.buffer) >= rl_config.batch_size:
            experiences = trainer.agent.buffer.sample(rl_config.batch_size)
            update_metrics = trainer.agent.update(experiences)
            training_metrics["policy_losses"].append(update_metrics.get("policy_loss", 0.0))
        
        # Update progress
        progress_bar.progress((episode + 1) / num_episodes)
        
        # Store model every 100 episodes
        if (episode + 1) % 100 == 0:
            st.session_state.current_model = trainer.policy_network.state_dict()
            st.session_state.current_config = config
    
    trainer.close()
    return training_metrics


def generate_text_sample(model, config: dict, prompt: str = None, num_samples: int = 5):
    """Generate text samples using the trained model."""
    env_config = NLGConfig(**config['env'])
    vocab = load_vocabulary()
    
    samples = []
    for _ in range(num_samples):
        if prompt:
            prompt_indices = vocab.encode(prompt)
            if prompt_indices:
                prompt_tensor = torch.tensor([prompt_indices], dtype=torch.long)
            else:
                prompt_tensor = torch.tensor([[0]], dtype=torch.long)
        else:
            prompt_tensor = torch.tensor([[0]], dtype=torch.long)
        
        with torch.no_grad():
            generated = model.generate(
                prompt=prompt_tensor,
                max_length=env_config.max_length,
                temperature=0.8,
                do_sample=True
            )
        
        generated_text = vocab.decode(generated[0].tolist())
        samples.append(generated_text)
    
    return samples


def plot_training_metrics(metrics: dict):
    """Create interactive plots for training metrics."""
    fig = go.Figure()
    
    # Episode rewards
    fig.add_trace(go.Scatter(
        y=metrics["episode_rewards"],
        mode='lines',
        name='Episode Rewards',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        title="Training Progress - Episode Rewards",
        xaxis_title="Episode",
        yaxis_title="Reward",
        height=400
    )
    
    return fig


def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 RL Text Generation Demo</h1>', unsafe_allow_html=True)
    
    # Warning disclaimer
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Disclaimer:</strong> This is a research/educational demonstration. 
        The models are trained on synthetic data and are NOT suitable for production use 
        or real-world text generation tasks. Results are for educational purposes only.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.markdown("## Configuration")
    
    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "RL Algorithm",
        ["ppo", "reinforce"],
        index=0,
        help="Choose the reinforcement learning algorithm"
    )
    
    # Model type selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["transformer", "lstm"],
        index=0,
        help="Choose the neural network architecture"
    )
    
    # Reward function selection
    reward_type = st.sidebar.selectbox(
        "Reward Function",
        ["length", "coherence", "diversity"],
        index=0,
        help="Choose the reward function for training"
    )
    
    # Training parameters
    st.sidebar.markdown("### Training Parameters")
    num_episodes = st.sidebar.slider(
        "Number of Episodes",
        min_value=50,
        max_value=1000,
        value=200,
        step=50,
        help="Number of training episodes"
    )
    
    learning_rate = st.sidebar.slider(
        "Learning Rate",
        min_value=1e-5,
        max_value=1e-2,
        value=3e-4,
        step=1e-5,
        format="%.0e",
        help="Learning rate for the optimizer"
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Training", "📊 Results", "🎯 Generation", "ℹ️ About"])
    
    with tab1:
        st.markdown('<h2 class="section-header">Model Training</h2>', unsafe_allow_html=True)
        
        # Create configuration
        config = {
            'env': {
                'vocab_size': 1000,
                'max_length': 30,
                'min_length': 3,
                'reward_type': reward_type,
                'temperature': 1.0,
                'seed': 42
            },
            'rl': {
                'algorithm': algorithm,
                'learning_rate': learning_rate,
                'gamma': 0.99,
                'lambda_gae': 0.95,
                'ppo_clip_ratio': 0.2,
                'ppo_epochs': 4,
                'value_loss_coef': 0.5,
                'entropy_coef': 0.01,
                'max_grad_norm': 0.5,
                'batch_size': 32,
                'buffer_size': 5000,
                'update_frequency': 5,
                'temperature': 1.0,
                'device': 'auto'
            },
            'model': {
                'model_type': model_type,
                'hidden_size': 128,
                'd_model': 128,
                'nhead': 4,
                'num_layers': 2,
                'dim_feedforward': 512,
                'dropout': 0.1
            },
            'training': {
                'num_episodes': num_episodes,
                'eval_frequency': 50,
                'log_dir': 'demo_logs',
                'use_wandb': False,
                'wandb_project': 'rl-text-generation-demo'
            }
        }
        
        # Training button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                with st.spinner("Training in progress..."):
                    progress_bar = st.progress(0)
                    
                    # Train model
                    training_metrics = train_model(config, num_episodes, progress_bar)
                    
                    # Store results
                    st.session_state.training_history.append({
                        'timestamp': datetime.now(),
                        'config': config,
                        'metrics': training_metrics
                    })
                    
                    st.success("Training completed!")
                    
                    # Show final metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Final Reward", f"{training_metrics['episode_rewards'][-1]:.3f}")
                    with col2:
                        st.metric("Final Quality", f"{training_metrics['text_qualities'][-1]:.3f}")
                    with col3:
                        st.metric("Avg Length", f"{np.mean(training_metrics['episode_lengths']):.1f}")
                    with col4:
                        st.metric("Episodes", len(training_metrics['episode_rewards']))
    
    with tab2:
        st.markdown('<h2 class="section-header">Training Results</h2>', unsafe_allow_html=True)
        
        if st.session_state.training_history:
            # Select training run
            training_runs = [f"Run {i+1} - {run['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}" 
                           for i, run in enumerate(st.session_state.training_history)]
            
            selected_run_idx = st.selectbox("Select Training Run", range(len(training_runs)), 
                                          format_func=lambda x: training_runs[x])
            
            selected_run = st.session_state.training_history[selected_run_idx]
            metrics = selected_run['metrics']
            
            # Plot training curves
            fig = plot_training_metrics(metrics)
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics summary
            st.markdown("### Training Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Episode Rewards**")
                st.write(f"Final: {metrics['episode_rewards'][-1]:.3f}")
                st.write(f"Average: {np.mean(metrics['episode_rewards']):.3f}")
                st.write(f"Max: {np.max(metrics['episode_rewards']):.3f}")
            
            with col2:
                st.markdown("**Text Quality**")
                st.write(f"Final: {metrics['text_qualities'][-1]:.3f}")
                st.write(f"Average: {np.mean(metrics['text_qualities']):.3f}")
                st.write(f"Max: {np.max(metrics['text_qualities']):.3f}")
            
            # Configuration used
            st.markdown("### Configuration Used")
            st.json(selected_run['config'])
            
        else:
            st.info("No training runs available. Please train a model first.")
    
    with tab3:
        st.markdown('<h2 class="section-header">Text Generation</h2>', unsafe_allow_html=True)
        
        if st.session_state.current_model is not None:
            # Generation parameters
            col1, col2 = st.columns(2)
            
            with col1:
                prompt = st.text_input("Prompt (optional)", placeholder="Enter a starting word or phrase")
                num_samples = st.slider("Number of samples", 1, 10, 5)
            
            with col2:
                temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
                max_length = st.slider("Max length", 5, 50, 20)
            
            # Generate button
            if st.button("🎯 Generate Text", type="primary"):
                with st.spinner("Generating text..."):
                    # Create model
                    model = create_model(st.session_state.current_config)
                    model.load_state_dict(st.session_state.current_model)
                    model.eval()
                    
                    # Generate samples
                    samples = generate_text_sample(model, st.session_state.current_config, 
                                                prompt, num_samples)
                    
                    # Display samples
                    st.markdown("### Generated Text Samples")
                    for i, sample in enumerate(samples, 1):
                        st.markdown(f"**Sample {i}:** {sample}")
                    
                    # Store for history
                    st.session_state.generated_texts.extend(samples)
            
            # Generation history
            if st.session_state.generated_texts:
                st.markdown("### Generation History")
                for i, text in enumerate(st.session_state.generated_texts[-10:], 1):
                    st.markdown(f"{i}. {text}")
                
                if st.button("Clear History"):
                    st.session_state.generated_texts = []
                    st.rerun()
        
        else:
            st.info("No trained model available. Please train a model first.")
    
    with tab4:
        st.markdown('<h2 class="section-header">About This Demo</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Overview
        This interactive demo showcases Reinforcement Learning (RL) for Natural Language Generation (NLG).
        The system trains RL agents to generate text by optimizing for various reward functions.
        
        ### Features
        - **Multiple RL Algorithms**: REINFORCE and PPO
        - **Different Model Architectures**: Transformer and LSTM
        - **Various Reward Functions**: Length, coherence, and diversity
        - **Interactive Training**: Real-time training with progress tracking
        - **Text Generation**: Generate samples with trained models
        - **Visualization**: Training curves and metrics
        
        ### How It Works
        1. **Environment**: A text generation environment with a vocabulary of common English words
        2. **Agent**: RL agent (REINFORCE or PPO) that learns to generate text
        3. **Reward Function**: Evaluates generated text based on length, coherence, or diversity
        4. **Training**: Agent learns through trial and error to maximize rewards
        5. **Generation**: Trained agent generates new text samples
        
        ### Technical Details
        - **Framework**: PyTorch for neural networks
        - **Environment**: Custom Gymnasium-compatible NLG environment
        - **Algorithms**: Policy gradient methods (REINFORCE, PPO)
        - **Models**: Transformer and LSTM architectures
        - **Evaluation**: Multiple quality metrics (coherence, diversity, length)
        
        ### Educational Purpose
        This demo is designed for educational and research purposes to understand:
        - How RL can be applied to text generation
        - The differences between various RL algorithms
        - The impact of different reward functions
        - The trade-offs between different model architectures
        
        ### Limitations
        - Trained on synthetic vocabulary (not real text data)
        - Simple reward functions for demonstration
        - Limited vocabulary size (1000 words)
        - Not suitable for production use
        
        ### Safety Notice
        This is a research demonstration. The models are not trained on real-world data
        and should not be used for production text generation tasks.
        """)


if __name__ == "__main__":
    main()
