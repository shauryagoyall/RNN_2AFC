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
        # half = self.batch_size // 2
        # signs = torch.cat([
        #     torch.ones(half, device=self.device),
        #     -torch.ones(half, device=self.device)
        # ])
        # if self.batch_size % 2 == 1:
        #     extra = torch.randint(0, 2, (1,), device=self.device) * 2 - 1
        #     signs = torch.cat([signs, extra])
        # signs = signs[torch.randperm(self.batch_size, device=self.device)]
        #
        # # Sample random magnitudes
        # mags = torch.rand(self.batch_size, device=self.device) * (self.max_diff - self.min_diff) + self.min_diff
        # mean_diff = signs * mags  # expected signed difference
        #
        # # Allocate inputs and targets
        # x = torch.zeros(self.seq_len, self.batch_size, 3, device=self.device)
        # y = torch.zeros(self.seq_len, self.batch_size, dtype=torch.long, device=self.device)
        #
        # # Fixation epoch
        # x[:self.fix_dur, :, 0] = 1.0
        #
        # stim = torch.zeros(self.stim_dur, self.batch_size, 2, device=self.device)
        # stim[:, :, 0] = + mean_diff.unsqueeze(0)
        # stim[:, :, 1] = - mean_diff.unsqueeze(0)
        # # stim += self.noise * torch.randn_like(stim)
        # x[self.fix_dur:self.fix_dur + self.stim_dur, :, 1:] = stim
        #
        # # Compute labels based on full summed evidence: -1 if left > right, else +1
        # sum_left = stim[:, :, 0].sum(dim=0)
        # sum_right = stim[:, :, 1].sum(dim=0)
        # labels_list = [ -1 if sum_left[i] > sum_right[i] else 1 for i in range(self.batch_size) ]
        # labels = torch.tensor(labels_list, dtype=torch.long, device=self.device)
        #
        # start = self.fix_dur + self.stim_dur + self.delay_dur
        # y[start:start + self.decision_dur, :] = labels.unsqueeze(0).expand(self.decision_dur, -1)
        #
        # return x, y

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
        mean_diff = signs * mags  # expected signed difference per trial

        # Allocate inputs and targets
        x = torch.zeros(self.seq_len, self.batch_size, 3, device=self.device)
        y = torch.zeros(self.seq_len, self.batch_size, dtype=torch.long, device=self.device)

        # Fixation epoch
        x[:self.fix_dur, :, 0] = 1.0

        # Stimulus epoch with variable temporal profiles
        # 1) Sample random positive profiles of shape [stim_dur, batch_size]
        profiles = torch.rand(self.stim_dur, self.batch_size, device=self.device)
        # 2) Normalize each to sum to 1 over time
        profiles = profiles / profiles.sum(dim=0, keepdim=True)
        # 3) Build left/right stimulus so total evidence = 0.2 * mean_diff
        stim = torch.zeros(self.stim_dur, self.batch_size, 2, device=self.device)
        stim[:, :, 0] = profiles * (0.2 * mean_diff).unsqueeze(0)
        stim[:, :, 1] = profiles * (-0.2 * mean_diff).unsqueeze(0)
        # 4) Add noise
        stim += self.noise * torch.randn_like(stim)
        # Insert into input tensor
        x[self.fix_dur:self.fix_dur + self.stim_dur, :, 1:] = stim

        ###Compute labels based on full summed evidence: -1 if left > right, else +1
        sum_left = stim[:, :, 0].sum(dim=0)
        sum_right = stim[:, :, 1].sum(dim=0)
        labels_list = [ -1 if sum_left[i] > sum_right[i] else 1 for i in range(self.batch_size) ]
        labels = torch.tensor(labels_list, dtype=torch.long, device=self.device)

        start = self.fix_dur + self.stim_dur + self.delay_dur
        y[start:start + self.decision_dur, :] = labels.unsqueeze(0).expand(self.decision_dur, -1)
        return x, y


class TwoAFCGeneratorWithFixedPoints:
    """
    Two-alternative forced choice task generator with two decision cues for fixed-point validation.
    During decision epoch, a constant cue input drives network toward one of two fixed points.
    """

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
            decision_cue_amp: float = 0.5,
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
        self.decision_cue_amp = decision_cue_amp
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
        # Now 4 channels: 0=fixation, 1=left stim, 2=right stim, 3=decision cue
        x = torch.zeros(self.seq_len, self.batch_size, 4, device=self.device)
        y = torch.zeros(self.seq_len, self.batch_size, dtype=torch.long, device=self.device)

        # Fixation epoch
        x[:self.fix_dur, :, 0] = 1.0

        # Stimulus epoch
        stim = torch.zeros(self.stim_dur, self.batch_size, 2, device=self.device)
        stim[:, :, 0] = +1.0 * mean_diff.unsqueeze(0)
        stim[:, :, 1] = -1.0 * mean_diff.unsqueeze(0)
        stim += self.noise * torch.randn_like(stim)
        x[self.fix_dur:self.fix_dur + self.stim_dur, :, 1:3] = stim

        # Compute decision labels based on summed evidence
        sum_left = stim[:, :, 0].sum(dim=0)
        sum_right = stim[:, :, 1].sum(dim=0)
        # Label: 0 for left choice, 1 for right choice
        labels = (sum_right > sum_left).long()

        # Delay epoch: no input except fixation off
        # Already zeros by default

        # Decision epoch: inject constant cue toward chosen fixed point
        start = self.fix_dur + self.stim_dur + self.delay_dur
        # Decision cue channel: +amp for right, -amp for left
        cue = (labels * 2 - 1).float() * self.decision_cue_amp
        x[start:start + self.decision_dur, :, 3] = cue.unsqueeze(0)

        # Targets: hold label during decision epoch (0 or 1)
        y[start:start + self.decision_dur, :] = labels.unsqueeze(0).expand(self.decision_dur, -1)

        return x, y
