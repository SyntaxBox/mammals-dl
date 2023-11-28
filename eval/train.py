# from tqdm.auto import tqdm
# import torch
# from torch import nn
# # import train_step
# # from test_step import test_step

# # 1. Take in various parameters required for training and test steps


# def train(model: torch.nn.Module,
#           train_dataloader: torch.utils.data.DataLoader,
#           test_dataloader: torch.utils.data.DataLoader,
#           optimizer: torch.optim.Optimizer,
#           loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
#           epochs: int = 5):

#     # 2. Create empty results dictionary
#     results = {"train_loss": [],
#                "train_acc": [],
#                "test_loss": [],
#                "test_acc": []
#                }

#     # 3. Loop through training and testing steps for a number of epochs
#     for epoch in tqdm(range(epochs)):
#         train_loss, train_acc = train_step(model=model,
#                                            dataloader=train_dataloader,
#                                            loss_fn=loss_fn,
#                                            optimizer=optimizer)
#         test_loss, test_acc = test_step(model=model,
#                                         dataloader=test_dataloader,
#                                         loss_fn=loss_fn)

#         # 4. Print out what's happening
#         print(
#             f"Epoch: {epoch+1} | "
#             f"train_loss: {train_loss:.4f} | "
#             f"train_acc: {train_acc:.4f} | "
#             f"test_loss: {test_loss:.4f} | "
#             f"test_acc: {test_acc:.4f}"
#         )

#         # 5. Update results dictionary
#         results["train_loss"].append(train_loss)
#         results["train_acc"].append(train_acc)
#         results["test_loss"].append(test_loss)
#         results["test_acc"].append(test_acc)

#     # 6. Return the filled results at the end of the epochs
#     return results
