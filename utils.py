import os
import random

import numpy as np
import torch


def resolve_device(*, require_cuda: bool = False) -> torch.device:
    """選用第一張可見 GPU，無 CUDA 時回傳 CPU。

    Args:
        require_cuda: 為 True 時，無 CUDA 即回報錯誤。
    Returns:
        模型與輸入共用的單一裝置。
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    if require_cuda:
        raise RuntimeError("Evaluation requires an available CUDA GPU")
    return torch.device("cpu")


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
