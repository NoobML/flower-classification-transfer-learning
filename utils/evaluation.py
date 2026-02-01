import torch
from sklearn.metrics import f1_score


def calculate_f1_score(model, test_loader, device):
    """Calculate macro F1-score on test set"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    return f1


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set (accuracy + F1)"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels.cpu()).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='macro')

    return accuracy, f1