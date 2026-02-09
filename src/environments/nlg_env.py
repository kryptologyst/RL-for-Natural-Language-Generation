"""Natural Language Generation Environment for Reinforcement Learning.

This module implements a Gymnasium-compatible environment for training RL agents
to generate natural language text using various reward functions.
"""

import gymnasium as gym
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import random
from collections import defaultdict


@dataclass
class NLGConfig:
    """Configuration for the NLG environment."""
    
    vocab_size: int = 1000
    max_length: int = 50
    min_length: int = 5
    reward_type: str = "length"  # "length", "coherence", "diversity", "custom"
    temperature: float = 1.0
    seed: Optional[int] = None


class Vocabulary:
    """Simple vocabulary for text generation."""
    
    def __init__(self, vocab_size: int = 1000, seed: Optional[int] = None):
        """Initialize vocabulary with common English words.
        
        Args:
            vocab_size: Size of the vocabulary
            seed: Random seed for reproducibility
        """
        self.vocab_size = vocab_size
        self.seed = seed
        
        # Common English words for demonstration
        common_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "up", "about", "into", "through", "during", "before", "after", "above",
            "below", "between", "among", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "must", "can", "shall", "this", "that", "these", "those",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
            "my", "your", "his", "her", "its", "our", "their", "mine", "yours", "hers",
            "ours", "theirs", "who", "what", "when", "where", "why", "how", "which", "whose",
            "cat", "dog", "house", "car", "tree", "water", "food", "book", "music", "love",
            "happy", "sad", "big", "small", "good", "bad", "new", "old", "hot", "cold",
            "fast", "slow", "beautiful", "ugly", "strong", "weak", "smart", "dumb", "kind",
            "mean", "friend", "family", "work", "play", "learn", "teach", "help", "give",
            "take", "make", "find", "see", "hear", "feel", "think", "know", "understand",
            "remember", "forget", "believe", "hope", "dream", "wish", "want", "need", "like",
            "dislike", "enjoy", "hate", "fear", "worry", "care", "try", "fail", "succeed",
            "win", "lose", "start", "stop", "begin", "end", "finish", "continue", "go",
            "come", "stay", "leave", "move", "run", "walk", "sit", "stand", "sleep", "wake",
            "eat", "drink", "cook", "clean", "wash", "dress", "wear", "buy", "sell", "pay",
            "cost", "spend", "save", "earn", "money", "time", "day", "night", "morning",
            "evening", "week", "month", "year", "today", "yesterday", "tomorrow", "now",
            "then", "here", "there", "everywhere", "somewhere", "nowhere", "always",
            "never", "sometimes", "often", "usually", "rarely", "again", "once", "twice",
            "first", "last", "next", "previous", "same", "different", "other", "another",
            "each", "every", "all", "some", "many", "few", "several", "most", "least",
            "more", "less", "much", "little", "enough", "too", "very", "quite", "rather",
            "pretty", "really", "truly", "actually", "finally", "suddenly", "quickly",
            "slowly", "carefully", "easily", "hardly", "almost", "nearly", "exactly",
            "probably", "possibly", "certainly", "definitely", "maybe", "perhaps",
            "surely", "obviously", "clearly", "apparently", "evidently", "naturally",
            "unfortunately", "fortunately", "luckily", "unluckily", "hopefully",
            "thankfully", "surprisingly", "interestingly", "importantly", "seriously",
            "honestly", "frankly", "basically", "generally", "usually", "normally",
            "typically", "commonly", "frequently", "occasionally", "rarely", "seldom",
            "hardly", "barely", "scarcely", "almost", "nearly", "quite", "rather",
            "pretty", "fairly", "somewhat", "slightly", "partially", "completely",
            "totally", "entirely", "fully", "absolutely", "perfectly", "exactly",
            "precisely", "accurately", "correctly", "properly", "appropriately",
            "suitably", "adequately", "sufficiently", "enough", "plenty", "lots",
            "loads", "tons", "heaps", "masses", "piles", "stacks", "bundles",
            "groups", "sets", "pairs", "couples", "dozens", "hundreds", "thousands",
            "millions", "billions", "trillions", "infinite", "endless", "limitless",
            "boundless", "countless", "innumerable", "numerous", "multiple", "various",
            "diverse", "different", "distinct", "separate", "individual", "single",
            "unique", "special", "particular", "specific", "certain", "definite",
            "clear", "obvious", "apparent", "evident", "visible", "noticeable",
            "remarkable", "outstanding", "exceptional", "extraordinary", "amazing",
            "incredible", "unbelievable", "fantastic", "wonderful", "marvelous",
            "brilliant", "excellent", "perfect", "ideal", "flawless", "faultless",
            "impeccable", "superb", "magnificent", "gorgeous", "stunning", "breathtaking",
            "spectacular", "phenomenal", "remarkable", "notable", "significant",
            "important", "crucial", "vital", "essential", "necessary", "required",
            "mandatory", "compulsory", "obligatory", "indispensable", "fundamental",
            "basic", "primary", "main", "principal", "chief", "leading", "top",
            "highest", "maximum", "supreme", "ultimate", "final", "last", "end",
            "conclusion", "finish", "completion", "termination", "stop", "halt",
            "pause", "break", "rest", "relaxation", "peace", "calm", "quiet",
            "silence", "stillness", "tranquility", "serenity", "harmony", "balance",
            "equilibrium", "stability", "security", "safety", "protection", "shelter",
            "refuge", "sanctuary", "haven", "home", "house", "building", "structure",
            "construction", "creation", "formation", "development", "growth",
            "progress", "advancement", "improvement", "enhancement", "upgrade",
            "betterment", "refinement", "perfection", "excellence", "quality",
            "standard", "level", "degree", "extent", "scope", "range", "scale",
            "measure", "size", "dimension", "proportion", "ratio", "percentage",
            "fraction", "part", "portion", "section", "segment", "piece", "bit",
            "fragment", "chunk", "block", "unit", "element", "component", "ingredient",
            "material", "substance", "matter", "stuff", "thing", "object", "item",
            "product", "goods", "merchandise", "commodity", "article", "piece",
            "sample", "specimen", "example", "instance", "case", "situation",
            "circumstance", "condition", "state", "status", "position", "place",
            "location", "spot", "point", "site", "area", "region", "zone", "territory",
            "country", "nation", "state", "province", "city", "town", "village",
            "community", "society", "population", "people", "individuals", "persons",
            "humans", "beings", "creatures", "animals", "plants", "nature", "world",
            "earth", "planet", "universe", "space", "cosmos", "galaxy", "star",
            "sun", "moon", "sky", "cloud", "rain", "snow", "wind", "storm",
            "weather", "climate", "season", "spring", "summer", "autumn", "winter",
            "morning", "afternoon", "evening", "night", "dawn", "dusk", "sunrise",
            "sunset", "daylight", "darkness", "light", "shadow", "brightness",
            "dimness", "color", "hue", "shade", "tint", "tone", "bright", "dark",
            "light", "heavy", "thick", "thin", "wide", "narrow", "broad", "deep",
            "shallow", "high", "low", "tall", "short", "long", "brief", "quick",
            "slow", "fast", "rapid", "swift", "speedy", "instant", "immediate",
            "sudden", "gradual", "steady", "constant", "regular", "irregular",
            "normal", "abnormal", "typical", "atypical", "usual", "unusual",
            "common", "uncommon", "rare", "frequent", "occasional", "rare",
            "seldom", "hardly", "barely", "scarcely", "almost", "nearly",
            "quite", "rather", "pretty", "fairly", "somewhat", "slightly",
            "partially", "completely", "totally", "entirely", "fully", "absolutely",
            "perfectly", "exactly", "precisely", "accurately", "correctly",
            "properly", "appropriately", "suitably", "adequately", "sufficiently",
            "enough", "plenty", "lots", "loads", "tons", "heaps", "masses",
            "piles", "stacks", "bundles", "groups", "sets", "pairs", "couples",
            "dozens", "hundreds", "thousands", "millions", "billions", "trillions"
        ]
        
        # Create vocabulary from common words, padding with numbers if needed
        self.word_to_idx = {}
        self.idx_to_word = {}
        
        for i, word in enumerate(common_words[:vocab_size]):
            self.word_to_idx[word] = i
            self.idx_to_word[i] = word
            
        # Fill remaining slots with numbers
        for i in range(len(common_words), vocab_size):
            word = str(i)
            self.word_to_idx[word] = i
            self.idx_to_word[i] = word
            
        # Special tokens
        self.pad_token = "<PAD>"
        self.start_token = "<START>"
        self.end_token = "<END>"
        self.unk_token = "<UNK>"
        
        # Add special tokens
        special_tokens = [self.pad_token, self.start_token, self.end_token, self.unk_token]
        for token in special_tokens:
            if token not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[token] = idx
                self.idx_to_word[idx] = token
                
        self.vocab_size = len(self.word_to_idx)
        
    def encode(self, text: str) -> List[int]:
        """Encode text to indices."""
        words = text.lower().split()
        indices = []
        for word in words:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                indices.append(self.word_to_idx[self.unk_token])
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """Decode indices to text."""
        words = []
        for idx in indices:
            if idx in self.idx_to_word:
                word = self.idx_to_word[idx]
                if word not in ["<PAD>", "<START>", "<END>", "<UNK>"]:
                    words.append(word)
        return " ".join(words)
    
    def get_random_word(self) -> int:
        """Get a random word index."""
        return random.randint(0, self.vocab_size - 1)


class NLGEnvironment(gym.Env):
    """Natural Language Generation Environment for RL training."""
    
    def __init__(self, config: NLGConfig):
        """Initialize the NLG environment.
        
        Args:
            config: Configuration for the environment
        """
        super().__init__()
        
        self.config = config
        self.vocab = Vocabulary(config.vocab_size, config.seed)
        
        # Set random seeds
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
            
        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(self.vocab.vocab_size)
        self.observation_space = gym.spaces.Box(
            low=0, high=self.vocab.vocab_size - 1, 
            shape=(config.max_length,), dtype=np.int32
        )
        
        # State variables
        self.current_text: List[int] = []
        self.step_count = 0
        self.done = False
        
        # Reward function
        self.reward_function = self._get_reward_function(config.reward_type)
        
    def _get_reward_function(self, reward_type: str):
        """Get the appropriate reward function."""
        if reward_type == "length":
            return self._length_reward
        elif reward_type == "coherence":
            return self._coherence_reward
        elif reward_type == "diversity":
            return self._diversity_reward
        else:
            return self._length_reward
            
    def _length_reward(self, text: List[int]) -> float:
        """Reward based on text length."""
        return min(len(text) / self.config.max_length, 1.0)
    
    def _coherence_reward(self, text: List[int]) -> float:
        """Reward based on text coherence (simplified)."""
        if len(text) < 2:
            return 0.0
            
        # Simple coherence: reward for common word transitions
        coherence_score = 0.0
        for i in range(len(text) - 1):
            # Check if current word is a common word
            current_word = self.vocab.idx_to_word.get(text[i], "")
            next_word = self.vocab.idx_to_word.get(text[i + 1], "")
            
            # Simple heuristic: reward common word combinations
            if current_word in ["the", "a", "an"] and next_word not in ["the", "a", "an"]:
                coherence_score += 0.1
            elif current_word in ["is", "are", "was", "were"] and next_word not in ["the", "a", "an"]:
                coherence_score += 0.1
                
        return min(coherence_score, 1.0)
    
    def _diversity_reward(self, text: List[int]) -> float:
        """Reward based on vocabulary diversity."""
        if len(text) == 0:
            return 0.0
            
        unique_words = len(set(text))
        total_words = len(text)
        return unique_words / total_words if total_words > 0 else 0.0
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.current_text = []
        self.step_count = 0
        self.done = False
        
        # Create initial observation (padded with zeros)
        obs = np.zeros(self.config.max_length, dtype=np.int32)
        
        info = {
            "text": "",
            "step_count": self.step_count,
            "done": self.done
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        if self.done:
            raise ValueError("Episode is done. Call reset() to start a new episode.")
            
        # Add action to current text
        self.current_text.append(action)
        self.step_count += 1
        
        # Calculate reward
        reward = self.reward_function(self.current_text)
        
        # Check termination conditions
        terminated = (
            len(self.current_text) >= self.config.max_length or
            action == self.vocab.word_to_idx[self.vocab.end_token] or
            self.step_count >= self.config.max_length
        )
        
        truncated = len(self.current_text) >= self.config.max_length
        
        self.done = terminated or truncated
        
        # Create observation
        obs = np.zeros(self.config.max_length, dtype=np.int32)
        obs[:len(self.current_text)] = self.current_text[:self.config.max_length]
        
        # Decode text for info
        decoded_text = self.vocab.decode(self.current_text)
        
        info = {
            "text": decoded_text,
            "step_count": self.step_count,
            "done": self.done,
            "reward": reward,
            "text_length": len(self.current_text)
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render the current state."""
        if mode == "human":
            text = self.vocab.decode(self.current_text)
            print(f"Generated text: {text}")
            return text
        elif mode == "rgb_array":
            # For visualization purposes
            return None
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def close(self):
        """Close the environment."""
        pass
