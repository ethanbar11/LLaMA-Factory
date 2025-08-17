#!/usr/bin/env python3

import json
import random
import os

def split_dataset(input_file, train_ratio=0.9):
    """
    Split a JSON dataset into training and test files.
    
    Args:
        input_file (str): Path to the input JSON file
        train_ratio (float): Ratio of data for training (default: 0.9)
    """
    
    # Read the JSON file
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total entries: {len(data)}")
    
    # Shuffle the data to ensure random distribution
    random.shuffle(data)
    
    # Calculate split point
    split_point = int(len(data) * train_ratio)
    
    # Split the data
    train_data = data[:split_point]
    test_data = data[split_point:]
    
    print(f"Training entries: {len(train_data)}")
    print(f"Test entries: {len(test_data)}")
    
    # Create output filenames
    base_name = os.path.splitext(input_file)[0]
    train_file = f"{base_name}_train.json"
    test_file = f"{base_name}_test.json"
    
    # Write training file
    print(f"Writing training data to {train_file}...")
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    # Write test file
    print(f"Writing test data to {test_file}...")
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print("Split completed successfully!")
    print(f"Training file: {train_file}")
    print(f"Test file: {test_file}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Input file path
    input_file = "my_data/omnisvg_only_less_than_1024_tokens.json"
    
    # Split the dataset
    split_dataset(input_file)
