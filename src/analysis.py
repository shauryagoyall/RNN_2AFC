import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def run_trial_and_collect_states(model, inputs):
    with torch.no_grad():
        seq_len, _ = inputs.size()
        h = torch.zeros(1, model.hidden_size, device=inputs.device)
        states = []
        for t in range(seq_len):
            x_t = inputs[t:t + 1]
            h = (1 - model.alpha) * h + model.alpha * (
                torch.relu(h @ model.W_rec.t() + x_t @ model.W_in + model.b_rec)
            )
            states.append(h.squeeze(0).cpu().numpy())
    return np.stack(states)


def plot_pca_trajectories(model, data_gen, n_trials=5, n_components=2):
    """
    Plots PCA of hidden states with color coded by choice and correctness.
    """
    device = next(model.parameters()).device
    all_states = []
    labels = []
    legend_tracker = set()
    correct_flags = []

    inputs, targets = data_gen()
    targets = targets[-1]  # assume final timestep contains decision target

    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.argmax(outputs[-1], dim=1)  # shape: [batch]

    for i in range(n_trials):
        inp = inputs[:, i, :].to(device)
        states = run_trial_and_collect_states(model, inp)
        all_states.append(states)
        targets[targets == -1] = 0
        pred = preds[i].item()
        target = targets[i].item()
        labels.append(pred)
        correct_flags.append(int(pred == target))

    # Fit PCA to all states
    data_mat = np.concatenate(all_states, axis=0)
    pca = PCA(n_components=n_components)
    pca.fit(data_mat)

    # Plot
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection='3d') if n_components == 3 else fig.add_subplot(111)

    for i, states in enumerate(all_states):
        coords = pca.transform(states)
        choice = labels[i]
        correct = correct_flags[i]

        if choice == 0:  # left
            color = 'tab:blue' if correct else 'tab:cyan'
        else:  # right
            color = 'tab:red' if correct else 'tab:pink'

        label = None
        key = (choice, correct)
        if key not in legend_tracker:
            label = f'Choice {choice} | {"Correct" if correct else "Incorrect"}'
            legend_tracker.add(key)

        if n_components == 3:
            ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color,
                    label=f'Trial {i} | Ch {choice} | {"✓" if correct else "✗"}')
        else:
            # ax.plot(coords[:, 0], coords[:, 1], color=color,
            #         label=f'Trial {i} | Ch {choice} | {"✓" if correct else "✗"}')
            ax.plot(coords[:, 0], coords[:, 1], color=color, label = label)


    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if n_components == 3:
        ax.set_zlabel("PC3")
    ax.set_title("PCA of Hidden-State Trajectories")
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.show()


def extract_states_and_labels(model, data_gen, n_batches=50):
    X_list, y_list = [], []
    device = next(model.parameters()).device
    model.eval()

    for _ in range(n_batches):
        inputs, targets = data_gen()
        inputs, targets = inputs.to(device), targets.to(device)
        seq_len = inputs.shape[0]

        h = torch.zeros(inputs.shape[1], model.hidden_size, device=device)
        for t in range(seq_len):
            x_t = inputs[t]
            h = (1 - model.alpha) * h + model.alpha * (
                torch.relu(h @ model.W_rec.t() + x_t @ model.W_in + model.b_rec)
            )
        X_list.append(h.detach().cpu().numpy())
        y_list.append(targets[-1].cpu().numpy())

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def decode_over_time(model, data_gen, n_batches=50):
    """
    Train separate logistic regressors at each timestep to decode choice.
    Returns:
        accuracies: list of decoding accuracies at each timestep
    """
    device = next(model.parameters()).device
    model.eval()

    # Collect all hidden states across batches
    all_states = []  # [n_batches * batch, seq_len, hidden_size]
    all_labels = []

    for _ in range(n_batches):
        inputs, targets = data_gen()
        seq_len, batch_size, _ = inputs.shape

        h = torch.zeros(batch_size, model.hidden_size, device=device)
        hidden_traj = []

        for t in range(seq_len):
            x_t = inputs[t]
            h = (1 - model.alpha) * h + model.alpha * (
                torch.relu(h @ model.W_rec.t() + x_t @ model.W_in + model.b_rec)
            )
            hidden_traj.append(h.detach().cpu().numpy())  # shape [batch, hidden]

        hidden_traj = np.stack(hidden_traj, axis=0).transpose(1, 0, 2)  # [batch, seq_len, hidden]
        all_states.append(hidden_traj)
        all_labels.append(targets[-1].cpu().numpy())  # shape [batch]

    X = np.concatenate(all_states, axis=0)  # [total_trials, seq_len, hidden]
    y = np.concatenate(all_labels, axis=0)  # [total_trials]

    seq_len = X.shape[1]
    accuracies = []

    for t in range(seq_len):
        X_t = X[:, t, :]  # [n_trials, hidden_size]
        clf = LogisticRegression(solver='lbfgs', max_iter=1000)
        scores = cross_val_score(clf, X_t, y, cv=5)
        accuracies.append(scores.mean())

    return accuracies


def plot_decoding_accuracy_over_time(decoder_accuracies, timepoints):
    plt.figure(figsize=(6, 4))
    plt.plot(timepoints, decoder_accuracies, marker='o')
    plt.axhline(0.5, color='gray', linestyle='--', label='Chance level')
    plt.ylim(0, 1)
    plt.xlabel("Time (steps)")
    plt.ylabel("Decoding Accuracy")
    plt.title("Logistic Regression Decoding Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=True):
    """
    Plot confusion matrix with optional normalization.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class label names (e.g. ["Left", "Right"])
        normalize: If True, normalize the matrix rows to sum to 1
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format=".2f" if normalize else "d")
    plt.title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
    plt.show()


def decode_choice(model, data_gen):
    """
    Train & cross-validate a linear decoder from hidden state → choice.
    """
    X, y = extract_states_and_labels(model, data_gen)
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"5-fold CV accuracy: {scores.mean() * 100:.1f}% ± {scores.std() * 100:.1f}%")

    clf.fit(X, y)
    weights = clf.coef_[0]

    # Plot confusion matrix
    plot_confusion_matrix(y, clf.predict(X), class_names=["Left", "Right"])
    accuracies = decode_over_time(model, data_gen, n_batches=10)
    plot_decoding_accuracy_over_time(accuracies, timepoints=range(len(accuracies)))
    return weights