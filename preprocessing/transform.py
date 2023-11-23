from torchvision import datasets
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
# import matplotlib.pyplot as plt

data_path = Path("data")
image_path = data_path / 'dataset'
# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"


# Write transform for image
data_transform = transforms.Compose([
    # Resize the images to 128x128
    transforms.Resize(size=(128, 128)),
    # Flip the images randomly on the horizontal
    # p = probability of flip, 0.5 = 50% chance
    transforms.RandomHorizontalFlip(p=0.5),
    # Turn the image into a torch.Tensor
    # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
    transforms.ToTensor()
])

# Use ImageFolder to create dataset(s)
train_data = datasets.ImageFolder(root=train_dir,  # target folder of images
                                  # transforms to perform on data (images)
                                  transform=data_transform,
                                  target_transform=None)  # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

# print(f"Train data:\n{train_data}\nTest data:\n{test_data}")

# Get class names as a list
class_names = train_data.classes
# print(class_names)
# Can also get class names as a dict
class_dict = train_data.class_to_idx
# print(class_dict)

# img, label = train_data[0][0], train_data[0][1]
# print(f"Image tensor:\n{img}")
# print(f"Image shape: {img.shape}")
# print(f"Image datatype: {img.dtype}")
# print(f"Image label: {label}")
# print(f"Label datatype: {type(label)}")
# # Rearrange the order of dimensions
# img_permute = img.permute(1, 2, 0)

# Print out different shapes (before and after permute)
# print(f"Original shape: {img.shape} -> [color_channels, height, width]")
# print(
#     f"Image permute shape: {img_permute.shape} -> [height, width, color_channels]")

# Plot the image
# plt.figure(figsize=(10, 7))
# plt.imshow(img.permute(1, 2, 0))
# plt.axis("off")
# plt.title(class_names[label], fontsize=14)
# plt.show()

# Turn train and test Datasets into DataLoaders
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=1,  # how many samples per batch?
                              # how many subprocesses to use for data loading? (higher = more)
                              num_workers=1,
                              shuffle=True)  # shuffle the data?

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=1,
                             num_workers=1,
                             shuffle=False)  # don't usually need to shuffle testing data

transforms = {class_names, class_dict, test_dataloader,
              train_dataloader, train_data, test_data}
