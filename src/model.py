import torch
import torch.nn as nn

class CTRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dt=1, tau=100.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.alpha = dt / tau
        # Recurrent weights and input/output projections
        self.W_rec = nn.Parameter(torch.randn(hidden_size, hidden_size) / hidden_size**0.5)
        self.W_in  = nn.Parameter(torch.randn(input_size, hidden_size) / input_size**0.5)
        self.b_rec = nn.Parameter(torch.zeros(hidden_size))
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, h = None):
        # inputs: seq_len × batch × input_size
        seq_len, batch, _ = inputs.size()
        if h is None:
            h = torch.zeros(batch, self.hidden_size, device=inputs.device)
        outputs = []
        for t in range(seq_len):
            x_t = inputs[t]
            u_t = h @ self.W_rec.t() + x_t @ self.W_in + self.b_rec
            h = (1 - self.alpha) * h + self.alpha * u_t
            h = torch.relu(h)
            outputs.append(self.fc_out(h))

        return torch.stack(outputs), h.detach()  # seq_len × batch × output_size