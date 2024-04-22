from pathlib import Path
import numpy as np
import pdb

data_audio = np.load("data/sampled_audio_data.npz", allow_pickle=True)["data"].item()
base_dir = Path("data/spokencoco")

for path in data_audio[0][:10]:
    ids = np.load(base_dir / (path.stem + ".npz"), allow_pickle=True)# ["ids"]
pdb.set_trace()