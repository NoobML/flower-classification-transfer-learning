import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style('whitegrid')


def plot_training_curves(results_dict, model_name, save_path=None):
    """Plot loss and accuracy curves for one model"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(results_dict['train_losses']) + 1)

    # Loss curves
    ax1.plot(epochs, results_dict['train_losses'], 'b-', linewidth=2, label='Train Loss')
    ax1.plot(epochs, results_dict['val_losses'], 'r-', linewidth=2, label='Val Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f"{model_name} - Loss Curves", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(epochs, results_dict['train_accs'], 'b-', linewidth=2, label='Train Acc')
    ax2.plot(epochs, results_dict['val_accs'], 'r-', linewidth=2, label='Val Acc')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f"{model_name} - Accuracy Curves", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_all_training_curves(all_results, save_path=None):
    """Plot training curves for all models in a grid"""
    n_models = len(all_results)
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 4 * n_models))

    if n_models == 1:
        axes = axes.reshape(1, -1)

    for idx, (model_name, results) in enumerate(all_results.items()):
        epochs = range(1, len(results['train_losses']) + 1)

        # Loss plot
        axes[idx, 0].plot(epochs, results['train_losses'], 'b-', linewidth=2, label='Train')
        axes[idx, 0].plot(epochs, results['val_losses'], 'r-', linewidth=2, label='Val')
        axes[idx, 0].set_ylabel('Loss', fontsize=11)
        axes[idx, 0].set_xlabel('Epoch', fontsize=11)
        axes[idx, 0].set_title(f"{model_name} - Loss", fontsize=12, fontweight='bold')
        axes[idx, 0].legend(fontsize=9)
        axes[idx, 0].grid(True, alpha=0.3)

        # Accuracy plot
        axes[idx, 1].plot(epochs, results['train_accs'], 'b-', linewidth=2, label='Train')
        axes[idx, 1].plot(epochs, results['val_accs'], 'r-', linewidth=2, label='Val')
        axes[idx, 1].set_ylabel('Accuracy (%)', fontsize=11)
        axes[idx, 1].set_xlabel('Epoch', fontsize=11)
        axes[idx, 1].set_title(f"{model_name} - Accuracy", fontsize=12, fontweight='bold')
        axes[idx, 1].legend(fontsize=9)
        axes[idx, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_model_comparison(all_results, save_path=None):
    """Compare all models: accuracy, parameters, training time"""
    models = list(all_results.keys())

    accuracies = [all_results[m]['best_val_acc'] for m in models]
    times = [all_results[m]['training_time'] for m in models]
    params = [all_results[m]['trainable_params'] / 1e6 for m in models]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy comparison
    bars1 = ax1.bar(models, accuracies, color='steelblue', edgecolor='black', linewidth=1.2)
    ax1.set_ylabel("Best Validation Accuracy (%)", fontsize=12)
    ax1.set_title("Model Accuracy Comparison", fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    # Training time comparison
    bars2 = ax2.bar(models, times, color='coral', edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Training Time (seconds)', fontsize=12)
    ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f"{height:.0f}s", ha='center', va='bottom', fontsize=9)

    # Parameters comparison
    bars3 = ax3.bar(models, params, color='lightgreen', edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('Trainable Parameters (Millions)', fontsize=12)
    ax3.set_title('Trainable Parameters Comparison', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}M', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()