"""
Plotting utilities.
"""
from pathlib import Path

import matplotlib.pyplot as plt


def plot_train_curves(
    epochs,
    train_acc,
    val_acc,
    train_loss,
    val_loss,
    model_name,
    dataset_name,
    output_dir,
    is_bsl,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Accuracy curve
    plt.figure()
    plt.plot(list(range(epochs)), train_acc, label='train')
    plt.plot(list(range(epochs)), val_acc, label='valid', ls='--')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.xticks()
    plt.yticks()
    plt.title(f'{model_name} acc')
    plt.legend()
    plt.grid(ls='--')
    plt.savefig(output_path / f'{model_name}_{dataset_name}_{'baseline' if is_bsl else 'kd'}_acc.png')

    # Loss curve
    plt.figure()
    plt.plot(list(range(epochs)), train_loss, label='train')
    plt.plot(list(range(epochs)), val_loss, label='valid', ls='--')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xticks()
    plt.yticks()
    plt.title(f'{model_name} loss')
    plt.legend()
    plt.grid(ls='--')
    plt.savefig(output_path / f'{model_name}_{dataset_name}_loss.png')
