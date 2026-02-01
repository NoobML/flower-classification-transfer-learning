import json
import os
import warnings

warnings.filterwarnings('ignore')

from utils import plot_training_curves, plot_all_training_curves, plot_model_comparison
import configs.config as config


def main():
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60 + "\n")

    # Load results
    try:
        with open(config.METRICS_FILE, 'r') as f:
            all_results = json.load(f)
        print(f"Loaded results for {len(all_results)} models\n")
    except FileNotFoundError:
        print(f"Error: {config.METRICS_FILE} not found!")
        print("Please run train_custom_cnn.py and train_pretrained.py first.")
        return

    # Create plots directory if it doesn't exist
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    # 1. Plot Custom CNN curves
    if 'Custom CNN' in all_results:
        print("1. Plotting Custom CNN training curves...")
        plot_training_curves(
            all_results['Custom CNN'],
            'Custom CNN',
            save_path=os.path.join(config.PLOTS_DIR, 'custom_cnn_curves.png')
        )

    # 2. Plot all models grid
    print("\n2. Plotting all models training curves...")
    plot_all_training_curves(
        all_results,
        save_path=os.path.join(config.PLOTS_DIR, 'all_models_grid.png')
    )

    # 3. Plot comparison
    print("\n3. Plotting model comparison...")
    plot_model_comparison(
        all_results,
        save_path=os.path.join(config.PLOTS_DIR, 'model_comparison.png')
    )

    # Summary
    print("\n" + "=" * 100)
    print("FINAL RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Model':<20} {'Best Val Acc':<15} {'F1-Score':<12} {'Train Time (s)':<16} {'Trainable Params':<18}")
    print("-" * 100)

    for model_name, results in all_results.items():
        acc = results['best_val_acc']
        f1 = results['f1_score']
        time_taken = results['training_time']
        params = results['trainable_params']
        print(f"{model_name:<20} {acc:>14.2f}% {f1:>11.4f} {time_taken:>15.1f} {params:>17,}")

    print("=" * 100)
    print(f"\nAll plots saved to: {config.PLOTS_DIR}")
    print("=" * 100)


if __name__ == '__main__':
    main()