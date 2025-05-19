import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from src.model import CTRNN
from src.data_generator import TwoAFCGenerator


def train_model(lr = 1e-3, num_epochs = 2000):
    # Hyperparameters
    input_size, hidden_size, output_size = 3, 128, 2
    steps = num_epochs

    model = CTRNN(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    criterion = nn.CrossEntropyLoss()
    data_gen = TwoAFCGenerator(batch_size = 32,
        fix_dur = 10,
        stim_dur = 20,
        delay_dur = 0,
        decision_dur = 10,
        min_diff= 0.1,
        max_diff= 0.6,
        noise_level= 0.2)

    device = torch.device("cpu")  # <-- use CPU temporarily
    model.to(device)

    epoch_loss = []
    epoch_accuracy = []
    start_time = time.time()
    for step in tqdm(range(steps)):
        inputs, targets = data_gen()
        targets[targets == -1] = 0
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)  # seq_len × batch × output
        # Compute loss only on decision period
        logits_dec = logits[data_gen.seq_len - data_gen.decision_dur:]
        targ_dec   = targets[data_gen.seq_len- data_gen.decision_dur:]
        loss = criterion(logits_dec.view(-1, output_size),
                         targ_dec.contiguous().view(-1))
        epoch_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits_dec.argmax(dim=-1)
            acc = (pred == targ_dec).float().mean().item()
            epoch_accuracy.append(acc)

    end_time = time.time()
    print("Time taken:", end_time - start_time)

    print(f"Step {step+1}/{steps} — Loss: {loss.item():.4f}, Acc: {acc*100:.1f}%")
    return model, epoch_loss, epoch_accuracy