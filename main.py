from utils.download_dataset import download_dataset
from preprocessing import train_test_split


def main():
    download_dataset()
    train_test_split()


if __name__ == "__main__":
    main()
