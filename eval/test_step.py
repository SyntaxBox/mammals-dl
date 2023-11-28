# import torch


# def test_step(model: torch.nn.Module,
#               dataloader: torch.utils.data.DataLoader,
#               loss_fn: torch.nn.Module):
#     # Put model in eval mode
#     model.eval()

#     # Setup test loss and test accuracy values
#     test_loss, test_acc = 0, 0

#     # Turn on inference context manager
#     with torch.inference_mode():
#         # Loop through DataLoader batches
#         for batch, (X, y) in enumerate(dataloader):
#             # Send data to target device

#             # 1. Forward pass
#             test_pred_logits = model(X)

#             # 2. Calculate and accumulate loss
#             loss = loss_fn(test_pred_logits, y)
#             test_loss += loss.item()

#             # Calculate and accumulate accuracy
#             test_pred_labels = test_pred_logits.argmax(dim=1)
#             test_acc += ((test_pred_labels == y).sum().item() /
#                          len(test_pred_labels))

#     # Adjust metrics to get average loss and accuracy per batch
#     test_loss = test_loss / len(dataloader)
#     test_acc = test_acc / len(dataloader)
#     return test_loss, test_acc
