import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile


def download_dataset():
    destination_folder = Path('./downloads')
    dataset = 'asaniczka/mammals-image-classification-dataset-45-animals'
    dataset_name = dataset.split("/")[-1]

    # Set your Kaggle API key
    api = KaggleApi()
    api.authenticate()

    # Check if the destination folder exists, and create it if not
    if not destination_folder.exists():
        destination_folder.mkdir(parents=True, exist_ok=True)

    # Check if the dataset is already downloaded
    dataset_folder = Path('./data')
    if not dataset_folder.exists():
        # Download the dataset
        api.dataset_download_files(dataset, path=str(destination_folder))

        # Unzip the downloaded files if needed
        # Adjust this based on the dataset format
        # with zipfile.ZipFile(destination_folder / f"{dataset}.zip", 'r') as zip_ref:
        #     zip_ref.extractall(dataset_folder)
        file_path = destination_folder / f"{dataset_name}.zip"
        if file_path.exists():
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_folder)
                print("Dataset download and extraction completed.")
        else:
            print(f"Error: File {file_path} not found.")

        # Remove the zip file if you want
        # os.remove(destination_folder / f"{dataset}.zip")

    else:
        print("Dataset files already exist.")
