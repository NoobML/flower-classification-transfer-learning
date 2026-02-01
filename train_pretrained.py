import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import json
import warnings

warnings.filterwarnings('ignore')

from models import get_pretrained_model
from utils import get_dataloaders, train_model, calculate_f1_score
import configs.config as config


def train_single_model(model_name, train_loader, val_loader, test_loader, device):
    """Train a single pretrained model"""
    print(f"\n{'=' * 50}")
    print(f"Training {model_name.upper()}")
    print(f"{'=' * 50}\n")

    # Load model
    model = get_pretrained_model(model_name, num_classes=config.NUM_CLASSES).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE
    )
    scheduler = StepLR(optimizer, step_size=config.STEP_SIZE_PRETRAINED, gamma=config.GAMMA)

    # Train
    train_losses, val_losses, train_accs, val_accs, training_time = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, config.PRETRAINED_EPOCHS
    )

    # Evaluate
    print("Calculating F1-score on test set...")
    f1 = calculate_f1_score(model, test_loader, device)
    print(f"F1-Score: {f1:.4f}\n")

    # Store results
    results = {
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

    return results


def main():
    print("=" * 60)
    print("TRAINING PRETRAINED MODELS")
    print("=" * 60 + "\n")

    # Setup
    device = config.DEVICE
    print(f"Using device: {device}\n")

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders()
    print()

    # Load existing results (Custom CNN)
    try:
        with open(config.METRICS_FILE, 'r') as f:
            all_results = json.load(f)
        print("Loaded existing results (Custom CNN)\n")
    except FileNotFoundError:
        all_results = {}
        print("No existing results found. Starting fresh.\n")

    # Train all models
    for model_name in config.PRETRAINED_MODELS:
        results = train_single_model(model_name, train_loader, val_loader, test_loader, device)
        all_results[model_name] = results

        # Save after each model
        with open(config.METRICS_FILE, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"Results saved for {model_name}\n")

    # Final summary
    print("\n" + "=" * 70)
    print("ALL MODELS TRAINED!")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Best Val Acc':<15} {'F1-Score':<12} {'Time (s)':<12}")
    print("-" * 70)

    for model_name, results in all_results.items():
        acc = results['best_val_acc']
        f1 = results['f1_score']
        time_taken = results['training_time']
        print(f"{model_name:<20} {acc:>14.2f}% {f1:>11.4f} {time_taken:>11.1f}")

    print("=" * 70)


if __name__ == '__main__':
    main()