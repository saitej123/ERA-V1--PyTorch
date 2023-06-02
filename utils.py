import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

train_losses = []
test_losses = []
train_acc = []
test_acc = []
test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}


def get_correct_pred_count(prediction, labels):
    """
    Calculates the number of correct predictions.

    Args:
        prediction (Tensor): Predicted labels.
        labels (Tensor): Ground truth labels.

    Returns:
        int: Number of correct predictions.
    """
    return prediction.argmax(dim=1).eq(labels).sum().item()

def train(model, device, train_loader, optimizer, train_acc):
    """
    Trains the model on the training dataset.

    Args:
        model (nn.Module): Model to train.
        device (torch.device): Device to run the training on.
        train_loader (DataLoader): DataLoader for the training dataset.
        optimizer (optim.Optimizer): Optimizer for model parameter updates.

    Returns:
        float: Training loss.
        float: Training accuracy.
    """
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0
    train_acc = []  # Initialize train_acc list

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = nn.functional.nll_loss(pred, target)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        correct += get_correct_pred_count(pred, target)
        processed += len(data)

        pbar.set_description(desc=f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)  # Append to train_acc list
    train_losses.append(train_loss/len(train_loader))

    return train_loss/len(train_loader), train_acc[-1]


def test(model, device, test_loader,test_acc):
    """
    Evaluates the model on the test dataset.

    Args:
        model (nn.Module): Model to evaluate.
        device (torch.device): Device to run the evaluation on.
        test_loader (DataLoader): DataLoader for the test dataset.

    Returns:
        float: Test loss.
        float: Test accuracy.
    """
    model.eval()

    test_loss = 0
    correct = 0
    total_samples = 0
    test_acc = []  # Initialize test_acc list

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()

            # Get predictions and calculate accuracy
            _, predicted = torch.max(output, dim=1)
            correct += (predicted == target).sum().item()
            total_samples += target.size(0)

    test_loss /= total_samples
    test_accuracy = 100.0 * correct / total_samples
    test_acc.append(test_accuracy)  # Append to test_acc list
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, total_samples, test_accuracy))

    return test_loss, test_accuracy


#Plotting the Training and Testing Accuracy and Loss Plots
def loss_plots(train_losses,train_acc,test_losses,test_acc):    
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

