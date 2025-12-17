import tensorflow as tf
import tensorflow_datasets as tfds
import os
from pathlib import Path

# Define classes
CLASSES = ['cat', 'dog']

def download_dataset():
    """
    Download and prepare the Cats vs Dogs dataset.
    """
    # Create data directories
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    print("Downloading Cats vs Dogs dataset...")
    (ds_train, ds_test), ds_info = tfds.load(
        'cats_vs_dogs',
        split=['train[:80%]', 'train[80%:]'],
        with_info=True,
        as_supervised=True
    )
    
    print(f"Dataset info: {ds_info}")
    
    # Save raw data (optional, as tfds handles it)
    # For this example, we'll use tfds directly in training
    
    # Check balance
    train_labels = []
    for _, label in ds_train:
        train_labels.append(label.numpy())
    
    cat_count = sum(1 for l in train_labels if l == 0)
    dog_count = sum(1 for l in train_labels if l == 1)
    
    print(f"Training data balance: Cats: {cat_count}, Dogs: {dog_count}")
    
    return ds_train, ds_test

if __name__ == "__main__":
    download_dataset()