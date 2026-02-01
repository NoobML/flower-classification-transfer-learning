import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import json
import warnings

warnings.filterwarnings('ignore')

from models import CustomCNN
from utils import get_dataloaders, train_model, calculate_f1_score
import configs.config as config


def main():
    print("=" * 60)
    print("TRAINING CUSTOM CNN")
    print("=" * 60 + "\n")

    # Setup
    device = config.DEVICE
    print(f"Using device: {device}\n")

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders()
    print()

    # Create model
    print("Creating Custom CNN...")
    model = CustomCNN(num_classes=config.NUM_CLASSES).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=config.STEP_SIZE_CUSTOM, gamma=config.GAMMA)

    # Train
    print(f"Training for {config.CUSTOM_CNN_EPOCHS} epoch(s)...\n")
    train_losses, val_losses, train_accs, val_accs, training_time = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, config.CUSTOM_CNN_EPOCHS
    )

    # Evaluate
    print("Calculating F1-score on test set...")
    f1 = calculate_f1_score(model, test_loader, device)
    print(f"F1-Score: {f1:.4f}\n")

    # Store results
    results = {
        'Custom CNN': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'training_time': training_time,
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'best_val_acc': max(val_accs),
            'final_val_acc': val_accs[-1],
            'f1_score': f1
        }
    }

    # Save results
    with open(config.METRICS_FILE, 'w') as f:
        json.dump(results, f, indent=4)

    print("=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best Val Acc: {results['Custom CNN']['best_val_acc']:.2f}%")
    print(f"F1-Score: {f1:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Results saved to {config.METRICS_FILE}")
    print("=" * 60)


if __name__ == '__main__':
    main()