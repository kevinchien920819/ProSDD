import os
import random

import numpy as np
import torch


def set_random_seed(seed: int, deterministic: bool = True) -> None:
    """Fix random seeds for Python, NumPy and PyTorch (CPU + all CUDA devices).

    Args:
        seed: the seed value.
        deterministic: if True, force cuDNN to use deterministic algorithms
            (slower but reproducible). Set False to keep cuDNN autotuning.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
