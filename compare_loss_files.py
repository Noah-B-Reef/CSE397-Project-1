#!/usr/bin/env python3
"""
Compare two training logs (e.g. loss.out and loss_hard.out) produced by the
absorption PINN runs. The script plots log10(total loss) versus epoch for both
logs and prints a small summary table.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot log10(total loss) from two training logs."
    )
    parser.add_argument(
        "--soft-log",
        type=Path,
        default=Path("my_models/loss.out"),
        help="Path to the soft-constraint training log (default: my_models/loss.out).",
    )
    parser.add_argument(
        "--hard-log",
        type=Path,
        default=Path("my_models/loss_hard.out"),
        help="Path to the hard-constraint training log (default: my_models/loss_hard.out).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("loss_comparison.png"),
        help="Destination for the generated plot.",
    )
    return parser.parse_args()


def parse_loss_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    history: List[tuple[int, float, float, float]] = []
    epoch = None
    latest_values = None

    with path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("Epoch:"):
                if epoch is not None and latest_values is not None:
                    history.append((epoch, *latest_values))
                epoch = int(line.split(":", 1)[1])
                latest_values = None
            elif line.startswith("final loss:"):
                numbers = re.findall(r"-?\d+\.\d+", line)
                if len(numbers) == 3:
                    latest_values = tuple(float(num) for num in numbers)
        if epoch is not None and latest_values is not None:
            history.append((epoch, *latest_values))

    if not history:
        raise ValueError(f"No epoch entries detected in {path}")

    return pd.DataFrame(
        history,
        columns=["epoch", "log10_total", "log10_supervised", "log10_residual"],
    )


def plot_losses(losses: Dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, df in losses.items():
        ax.plot(
            df["epoch"],
            df["log10_total"],
            label=label,
            linewidth=1.2,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("log10 total loss")
    ax.set_title("Loss per epoch comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")


def main() -> None:
    args = parse_arguments()

    losses = {
        "soft": parse_loss_log(args.soft_log),
        "hard": parse_loss_log(args.hard_log),
    }

    summary_rows = []
    for label, df in losses.items():
        summary_rows.append(
            {
                "model": label,
                "epochs_logged": int(df["epoch"].iloc[-1]) + 1,
                "final_log10_total": df["log10_total"].iloc[-1],
                "final_log10_supervised": df["log10_supervised"].iloc[-1],
                "final_log10_residual": df["log10_residual"].iloc[-1],
            }
        )

    summary = pd.DataFrame(summary_rows)
    print("\nLoss summary:")
    print(summary.to_string(index=False))

    plot_losses(losses, args.output)


if __name__ == "__main__":
    main()

