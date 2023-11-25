import torch
from utils.download_dataset import download_dataset
from preprocessing.train_test_split import train_test_split
from models.image_classification_V0 import ImageModelV0
from preprocessing.transform import transforms
from eval.single_image_eval import eval

torch.manual_seed(42)


def main():
    download_dataset()
    train_test_split()
    class_names, class_dict, test_dataloader, train_dataloader, train_data, test_data = transforms
    label, img = next(iter(test_dataloader))
    print(label.shape, img.shape)
    model_0 = ImageModelV0(input_shape=3, hidden_units=128,
                           output_shape=len(class_names))
    eval(data_loader=train_dataloader, model=model_0)


if __name__ == "__main__":
    main()
