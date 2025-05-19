import matplotlib.pyplot as plt
import numpy as np

def plot_two_afc_batches(generator, num_trials=3):
    """
    Plot inputs and targets for a few trials from a TwoAFCGenerator.

    Args:
        generator: an instance of TwoAFCGenerator
        num_trials: how many trials (from batch) to plot
    """
    x, y = generator()
    seq_len, batch_size, _ = x.shape
    n = min(batch_size, num_trials)

    for idx in range(n):
        xi = x[:, idx, :].cpu().numpy()  # [time, 3]
        yi = y[:, idx].cpu().numpy()     # [time]

        target_L = (yi == -1).astype(float)
        target_R = (yi == 1).astype(float)

        x_plot = np.concatenate([xi, target_L[:, None], target_R[:, None]], axis=1).T

        fig, ax1 = plt.subplots(figsize=(8, 4))
        im = ax1.imshow(x_plot, aspect='auto', interpolation='nearest', cmap='viridis')

        ax1.set_yticks([0, 1, 2, 3, 4])
        ax1.set_yticklabels(['Fixation', 'Stim L', 'Stim R', 'Target L (-1)', 'Target R (1)'])

        ax1.set_xlabel('Time step')
        ax1.set_title(f'Trial {idx + 1}: Inputs and Target')

        plt.colorbar(im, ax=ax1, orientation='vertical', label='Value')
        plt.tight_layout()
        plt.show()




