from src.data_generator import TwoAFCGenerator, TwoAFCGeneratorWithFixedPoints
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit


def sigmoid(x, x0, k):
    """Standard logistic function."""
    return 1 / (1 + np.exp(-(x - x0) * k))


def compute_psychometric_curve(stim1, stim2, model_outputs, n_bins=10):
    """
    Computes and plots the psychometric curve with a sigmoid fit.

    Args:
        stim1, stim2: tensors of shape [batch] — per-trial stimulus values
        model_outputs: tensor of predicted class per trial (0 or 1)
        n_bins: number of bins to group stim differences
    """
    stim_diff = (stim2 - stim1).detach().numpy()
    choices = model_outputs.detach().cpu().numpy()

    bin_means, bin_edges, _ = binned_statistic(stim_diff, choices, statistic='mean', bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    try:
        [popt, _] = curve_fit(sigmoid, stim_diff, choices, p0=[0.0, 1.0], maxfev=5000)
        fit_x = np.linspace(min(stim_diff), max(stim_diff), 200)
        fit_y = sigmoid(fit_x, *popt)
    except RuntimeError:
        print("Sigmoid fit failed. Plotting binned data only.")
        fit_x, fit_y = None, None

    plt.figure(figsize=(6, 4))
    plt.plot(bin_centers, bin_means, 'o-', label='Model choice prob.')
    if fit_x is not None:
        plt.plot(fit_x, fit_y, 'r--', label='Sigmoid fit')
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel("Stimulus difference (stim2 - stim1)")
    plt.ylabel("P(choose stim2)")
    plt.title("Psychometric Curve with Sigmoid Fit")
    plt.ylim([-0.1, 1.1])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return

def plot_psychometric_curve(trained_model):
    gen = TwoAFCGenerator(batch_size = 1000,
            fix_dur = 10,
            stim_dur = 20,
            delay_dur = 0,
            decision_dur = 10,
            min_diff= 0,
            max_diff= 0.5,
            noise_level= 0.2)
    inputs, targets = gen()

    with (torch.no_grad()):
        outputs, _ = trained_model(inputs)           # [seq_len, batch, num_classes]
        final_outputs = outputs[-1]       # use final timestep
        pred = torch.argmax(final_outputs, dim=1)  # [batch] — predicted class index

    stim1 = torch.sum(inputs[:, :, 1], dim=0)
    stim2 = torch.sum(inputs[:, :, 2], dim=0)
    compute_psychometric_curve(stim1, stim2, pred, n_bins=20)

    return

def plot_learning_curves(train_losses):
    """
    Plots training (and optionally validation) loss and accuracy curves.

    Args:
        train_losses (list of float): loss at each logging step during training.
        train_accs   (list of float): accuracy (0–1) at each logging step during training.
        val_losses   (list of float, optional): loss at each logging step on validation set.
        val_accs     (list of float, optional): accuracy (0–1) on validation set.
    """
    steps = range(1, len(train_losses) + 1)

    # Loss curve
    plt.figure()
    plt.plot(steps, train_losses, label="Train Loss")
    plt.xlabel("Logging Step")
    plt.ylabel("Cross‐Entropy Loss")
    plt.yscale('log')
    plt.title("Loss Curve")
    plt.legend()
    plt.show()

