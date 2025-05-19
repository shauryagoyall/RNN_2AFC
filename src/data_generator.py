import torch

class TwoAFCGenerator:
    def __init__(
        self,
        batch_size: int = 32,
        fix_dur: int = 20,
        stim_dur: int = 15,
        delay_dur: int = 5,
        decision_dur: int = 10,
        min_diff: float = 0.2,
        max_diff: float = 1.0,
        noise_level: float = 0.1,
        device=None
    ):
        assert batch_size > 1, "batch_size must be at least 2"
        self.batch_size = batch_size
        self.fix_dur = fix_dur
        self.stim_dur = stim_dur
        self.delay_dur = delay_dur
        self.decision_dur = decision_dur
        self.seq_len = fix_dur + stim_dur + delay_dur + decision_dur
        self.min_diff = min_diff
        self.max_diff = max_diff
        self.noise = noise_level
        self.device = device or torch.device('cpu')

    def __call__(self):
        # Balance sign of mean difference across batch
        half = self.batch_size // 2
        signs = torch.cat([
            torch.ones(half, device=self.device),
            -torch.ones(half, device=self.device)
        ])
        if self.batch_size % 2 == 1:
            extra = torch.randint(0, 2, (1,), device=self.device) * 2 - 1
            signs = torch.cat([signs, extra])
        signs = signs[torch.randperm(self.batch_size, device=self.device)]

        # Sample random magnitudes
        mags = torch.rand(self.batch_size, device=self.device) * (self.max_diff - self.min_diff) + self.min_diff
        mean_diff = signs * mags  # expected signed difference

        # Allocate inputs and targets
        x = torch.zeros(self.seq_len, self.batch_size, 3, device=self.device)
        y = torch.zeros(self.seq_len, self.batch_size, dtype=torch.long, device=self.device)

        # Fixation epoch
        x[:self.fix_dur, :, 0] = 1.0

        stim = torch.zeros(self.stim_dur, self.batch_size, 2, device=self.device)
        stim[:, :, 0] = +0.2 * mean_diff.unsqueeze(0)
        stim[:, :, 1] = -0.2 * mean_diff.unsqueeze(0)
        stim += self.noise * torch.randn_like(stim)
        x[self.fix_dur:self.fix_dur + self.stim_dur, :, 1:] = stim

        # Compute labels based on full summed evidence: -1 if left > right, else +1
        sum_left = stim[:, :, 0].sum(dim=0)
        sum_right = stim[:, :, 1].sum(dim=0)
        labels_list = [ -1 if sum_left[i] > sum_right[i] else 1 for i in range(self.batch_size) ]
        labels = torch.tensor(labels_list, dtype=torch.long, device=self.device)

        start = self.fix_dur + self.stim_dur + self.delay_dur
        y[start:start + self.decision_dur, :] = labels.unsqueeze(0).expand(self.decision_dur, -1)

        return x, y