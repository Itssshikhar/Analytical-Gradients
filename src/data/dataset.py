"""
Subject-verb agreement dataset preparation.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from datasets import Dataset, DatasetDict
import torch
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class SVADataset:
    """
    Subject-verb agreement dataset preparation.
    
    This class handles loading, preprocessing, and tokenizing examples for 
    subject-verb agreement tasks.
    """
    
    def __init__(
        self, 
        tokenizer,
        max_length: int = 128,
        correct_only: bool = False,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            tokenizer: Tokenizer for the language model
            max_length: Maximum sequence length
            correct_only: If True, only include grammatically correct examples
            cache_dir: Directory to cache processed datasets
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.correct_only = correct_only
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def generate_sva_examples(self, num_examples: int = 200) -> List[Dict[str, Any]]:
        """
        Generate synthetic subject-verb agreement examples.
        
        Args:
            num_examples: Number of examples to generate
            
        Returns:
            List of examples with keys: text, label, subject_position, verb_position
        """
        singular_subjects = [
            "The man", "The woman", "The child", "The dog", "The cat",
            "A student", "The teacher", "The doctor", "A scientist", "The president",
            "My friend", "The engineer", "The artist", "The writer", "The musician"
        ]
        
        plural_subjects = [
            "The men", "The women", "The children", "The dogs", "The cats",
            "Some students", "The teachers", "The doctors", "Several scientists", "The presidents",
            "My friends", "The engineers", "The artists", "The writers", "The musicians"
        ]
        
        singular_verbs = [
            "walks", "runs", "eats", "sleeps", "works",
            "reads", "writes", "studies", "plays", "sings",
            "dances", "drives", "swims", "talks", "thinks"
        ]
        
        plural_verbs = [
            "walk", "run", "eat", "sleep", "work",
            "read", "write", "study", "play", "sing",
            "dance", "drive", "swim", "talk", "think"
        ]
        
        adverbial_phrases = [
            "every day",
            "in the park",
            "at school",
            "on weekends",
            "during the summer",
            "with enthusiasm",
            "very quickly",
            "for hours",
            "at night",
            "after lunch",
            ""  # Empty string for no adverbial phrase
        ]
        
        examples = []
        for _ in range(num_examples):
            # Decide if we're generating a singular or plural example
            is_singular = np.random.random() < 0.5
            
            # Select subject
            if is_singular:
                subject = np.random.choice(singular_subjects)
                correct_verb = np.random.choice(singular_verbs)
                incorrect_verb = np.random.choice(plural_verbs)
            else:
                subject = np.random.choice(plural_subjects)
                correct_verb = np.random.choice(plural_verbs)
                incorrect_verb = np.random.choice(singular_verbs)
            
            # Add an adverbial phrase
            adverbial = np.random.choice(adverbial_phrases)
            
            # Create correct and incorrect examples
            correct_text = f"{subject} {correct_verb} {adverbial}".strip()
            incorrect_text = f"{subject} {incorrect_verb} {adverbial}".strip()
            
            # Get token positions
            subject_end = len(subject.split())
            
            # Add to examples
            examples.append({
                "text": correct_text,
                "label": 1,  # 1 for correct
                "subject_position": (0, subject_end - 1),
                "verb_position": subject_end
            })
            
            if not self.correct_only:
                examples.append({
                    "text": incorrect_text,
                    "label": 0,  # 0 for incorrect
                    "subject_position": (0, subject_end - 1),
                    "verb_position": subject_end
                })
        
        return examples
    
    def prepare_dataset(self, num_examples: int = 200) -> Dataset:
        """
        Prepare the subject-verb agreement dataset.
        
        Args:
            num_examples: Number of examples to generate
            
        Returns:
            Dataset object with tokenized examples
        """
        cache_path = os.path.join(self.cache_dir, f"sva_dataset_{num_examples}.json") if self.cache_dir else None
        
        # Try to load from cache first
        if cache_path and os.path.exists(cache_path):
            logger.info(f"Loading dataset from cache: {cache_path}")
            with open(cache_path, 'r') as f:
                examples = json.load(f)
        else:
            logger.info(f"Generating {num_examples} subject-verb agreement examples")
            examples = self.generate_sva_examples(num_examples)
            
            # Save to cache if needed
            if cache_path:
                with open(cache_path, 'w') as f:
                    json.dump(examples, f)
        
        # Create a Hugging Face dataset
        dataset = Dataset.from_dict({
            "text": [ex["text"] for ex in examples],
            "label": [ex["label"] for ex in examples],
            "subject_position": [ex["subject_position"] for ex in examples],
            "verb_position": [ex["verb_position"] for ex in examples]
        })
        
        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=["text"]
        )
        
        return tokenized_dataset
    
    def get_train_test_split(self, num_examples: int = 200, test_size: float = 0.2) -> DatasetDict:
        """
        Get a train/test split of the dataset.
        
        Args:
            num_examples: Number of examples to generate
            test_size: Fraction of examples to use for testing
            
        Returns:
            DatasetDict with train and test splits
        """
        dataset = self.prepare_dataset(num_examples)
        
        # Create a train/test split
        splits = dataset.train_test_split(test_size=test_size)
        
        return splits 