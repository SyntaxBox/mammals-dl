from pathlib import Path
import os
import shutil
import random

DATA_PATH = Path('data')
MAMMALS_PATH = DATA_PATH / 'mammals'


def load_dataset():
    # Get a list of all image paths and their corresponding labels
    image_path_list = list(MAMMALS_PATH.glob("**/*.jpg"))
    return image_path_list


def train_test_split():
    # Create the dataset directory if it doesn't exist
    dataset_dir = DATA_PATH / 'dataset'
    if not dataset_dir.exists():
        dataset_dir.mkdir()
    else:
        print("Dataset already exists.")
        return

    # Load the list of image paths
    image_list = load_dataset()

    # Split the data into train and test sets
    for image_path in image_list:
        # Extract information from the path
        parts = list(image_path.parts)
        _, mammal_type, _, _ = parts[1], parts[2], parts[-2], parts[-1]

        # Create train and test directories for each mammal type
        train_dir = dataset_dir / 'train' / mammal_type
        test_dir = dataset_dir / 'test' / mammal_type

        # Create directories if they don't exist
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        # Decide whether the image goes to the train or test set (80% train, 20% test)
        if random.random() < 0.8:
            destination = train_dir
        else:
            destination = test_dir

        # Copy the image to the appropriate directory
        shutil.copy(image_path, destination / image_path.name)
