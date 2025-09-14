"""Utility for leveraging all available CUDA GPUs in PyTorch models.
This module exposes :class:`CudaMultiGPU`, a lightweight wrapper that detects
all CUDA devices on the host and moves a model to run on them in parallel via
``torch.nn.DataParallel``.  It can be imported and used in any project without
modification.
"""

from __future__ import annotations

import torch


class CudaMultiGPU:

    def __init__(self, model: torch.nn.Module, verbose: bool = True) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if torch.cuda.is_available():
            if self.device_count > 1:
                model = torch.nn.DataParallel(model, device_ids=list(range(self.device_count)))
            model = model.to(self.device)
            if verbose:
                print(f"Using {self.device_count} CUDA device(s).")
        elif verbose:
            print("CUDA not available. Running on CPU.")

        self.model = model

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


if __name__ == "__main__":  
    class _ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    example_model = _ToyModel()
    mgpu = CudaMultiGPU(example_model)
    sample_input = torch.randn(1, 4).to(mgpu.device)
    print(mgpu(sample_input))
