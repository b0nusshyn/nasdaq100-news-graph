"""Check this machine can run the model stages: torch + PyTorch Geometric (task 0.4, issue #4).

    python -m src.core.check_env

Prints versions and runs one tiny GAT forward pass. Post the output on issue #4.
"""

from __future__ import annotations

import platform
import sys


def check() -> dict[str, str]:
    import torch
    import torch_geometric
    from torch_geometric.nn import GATConv

    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]])
    out, (_, attention) = GATConv(4, 8, heads=2).forward(x, edge_index, return_attention_weights=True)
    assert out.shape == (3, 16), out.shape
    assert attention.shape[1] == 2, attention.shape

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else "not available (CPU only)",
        "torch_geometric": torch_geometric.__version__,
        "gat_forward": "ok",
    }


def main() -> None:
    import argparse

    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    try:
        info = check()
    except Exception as e:  # report any failure as a readable message, not a traceback
        print(f"FAILED: {type(e).__name__}: {e}")
        print("See README Setup; on failure, run the model stages on Colab.")
        sys.exit(1)
    for k, v in info.items():
        print(f"{k:16} {v}")


if __name__ == "__main__":
    main()
