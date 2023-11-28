import torch
from torch import nn
from utils.download_dataset import download_dataset
from preprocessing.train_test_split import train_test_split
from models.image_classification_V0 import ImageModelV0
from preprocessing.transform import transforms
from eval.eval import model_eval

torch.manual_seed(42)


def main():
    download_dataset()
    train_test_split()
    class_names, class_dict, test_dataloader, train_dataloader, train_data, test_data = transforms
    label, img = next(iter(test_dataloader))
    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    model_0 = ImageModelV0(input_shape=3, hidden_units=128,
                           output_shape=len(class_names))
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)
    result = model_eval(model=model_0,
                        loss_fn=loss_fn,
                        optimizer=optimizer, train_dataloader=train_dataloader, test_dataloader=test_dataloader, num_epochs=5)


if __name__ == "__main__":
    main()
