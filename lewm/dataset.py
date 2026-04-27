import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TetrisDataset(Dataset):
    """
    Loads trajectories saved by generate_trajectories.py.

    File layout:
        data_dir/{traj}-{step}-im.png    — frame BEFORE action at this step
        data_dir/{traj}-{step}-meta.json — {"action": int, "reward": float, ...}
        data_dir/{traj}-{T}-im.png       — terminal frame (no meta file)

    Action alignment:
        meta at step t contains the action taken FROM state t to reach state t+1.
        This matches LeWM's prediction: predictor(emb[t], actions[t]) → emb[t+1].

    Returns overlapping sub-trajectories of length seq_len:
        obs:     (seq_len, 3, img_size, img_size)  — float32, normalized
        actions: (seq_len,)                         — int64

    We load seq_len consecutive steps where all have a meta file (i.e. non-terminal).
    The last obs in a sub-trajectory has a target (the first obs of the NEXT sub-trajectory
    slot) because the LeWM loss only uses emb[:, 1:] as targets for next_emb[:, :-1].
    """

    def __init__(self, data_dir: str, seq_len: int = 4, img_size: int = 224):
        self.seq_len = seq_len
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        data_path = Path(data_dir)

        # Find all trajectory IDs
        traj_ids = sorted({
            int(p.stem.split("-")[0])
            for p in data_path.glob("*-im.png")
        })

        # Build index of valid sub-trajectories.
        # A step is "actionable" if it has a meta file (non-terminal).
        # We need seq_len actionable steps + one next-frame (for the last prediction target).
        self.index = []   # list of (traj_id, [step_0, step_1, ..., step_{seq_len-1}])

        for traj_id in traj_ids:
            actionable = sorted({
                int(p.stem.split("-")[1])
                for p in data_path.glob(f"{traj_id}-*-meta.json")
            })
            # Sliding window over actionable steps
            for i in range(len(actionable) - seq_len + 1):
                window = actionable[i : i + seq_len]
                # Require all steps are consecutive (no gaps from early game-over)
                if window[-1] - window[0] == seq_len - 1:
                    self.index.append((traj_id, window))

        self.data_path = data_path

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        traj_id, steps = self.index[idx]

        obs_list, action_list = [], []
        for step in steps:
            img = Image.open(
                self.data_path / f"{traj_id}-{step}-im.png"
            ).convert("RGB")
            obs_list.append(self.transform(img))

            with open(self.data_path / f"{traj_id}-{step}-meta.json") as f:
                action_list.append(json.load(f)["action"])

        obs = torch.stack(obs_list)
        actions = torch.tensor(action_list, dtype=torch.long)
        return obs, actions
