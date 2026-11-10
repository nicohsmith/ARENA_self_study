#%%
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter0_fundamentals"
repo = "ARENA_materials"
branch = "main"


# Install dependencies
try:
    import jaxtyping
except ImportError:
    print('need to pip install einops jaxtyping torchinfo wandb datasets')

if IN_COLAB:
    import subprocess

    chapter = "chapter0_fundamentals"
    repo = "ARENA_materials"
    root = "/content"

    if not Path(f"{root}/{chapter}").exists():
        subprocess.run(["sudo", "apt-get", "install", "unzip"])
        subprocess.run([sys.executable, "-m", "pip", "install", "jupyter", "ipython", "--upgrade"])
        subprocess.run(["wget", "-P", root, f"https://github.com/ARENA-education/ARENA_materials/archive/refs/heads/{branch}.zip"])
        subprocess.run(["unzip", f"{root}/{branch}.zip", f"{repo}-{branch}/{chapter}/exercises/*", "-d", root])
        subprocess.run(["mv", f"{root}/{repo}-{branch}/{chapter}", f"{root}/{chapter}"])
        subprocess.run(["rm", f"{root}/{branch}.zip"])
        subprocess.run(["rmdir", f"{root}/{repo}-{branch}"])

    exercises_dir = Path(f"{root}/{chapter}/exercises")
else:
    # Local clone: this file lives at <exercises_dir>/part3_optimization/exercises_multiple_gpus.py
    exercises_dir = Path(__file__).resolve().parent.parent

assert exercises_dir.exists(), (
    "Unexpected error: please manually clone ARENA repo"
)

if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

os.chdir(exercises_dir)
#%%
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

import numpy as np
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from IPython.core.display import HTML
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor, optim
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

# Make sure exercises are in the path
section = "part3_optimization"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))


MAIN = __name__ == "__main__"

import part3_optimization.tests as tests
from fundamentals_utils import IMAGENET_TRANSFORM, CIFAR10
from part2_cnns.solutions import Linear, ResNet34, get_resnet_for_feature_extraction
from part3_optimization.utils import plot_fn, plot_fn_with_points
from plotly_utils import bar, imshow, line

# Cap the number of CPU threads PyTorch uses. On machines with many cores the default of one
# thread per core makes small CPU tensor operations much slower
t.set_num_threads(min(4, t.get_num_threads()))

from IPython.display import display
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "vscode"
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

#%%
IN_COLAB = "google.colab" in sys.modules
assert not IN_COLAB, "Should be doing these exercises in VS Code"

#%%