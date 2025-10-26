import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


def parse_epoch_losses(loss_path: Path) -> pd.DataFrame:
    history = []
    epoch = None
    last_triplet = None

    with loss_path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("Epoch:"):
                if epoch is not None and last_triplet is not None:
                    history.append((epoch, *last_triplet))
                epoch = int(line.split(":", 1)[1])
                last_triplet = None
            elif line.startswith("final loss:"):
                numbers = re.findall(r"-?\d+\.\d+", line)
                if len(numbers) == 3:
                    last_triplet = tuple(float(num) for num in numbers)
        if epoch is not None and last_triplet is not None:
            history.append((epoch, *last_triplet))

    if not history:
        raise ValueError(f"No epoch entries detected in {loss_path}")

    return pd.DataFrame(
        history,
        columns=["epoch", "log10_total", "log10_supervised", "log10_residual"],
    )


def load_and_merge(log_paths: dict[str, Path]) -> pd.DataFrame:
    merged = None
    for name, path in log_paths.items():
        df = parse_epoch_losses(path)
        df = df[["epoch", "log10_total"]].rename(
            columns={"log10_total": f"log10_total_{name}"}
        )
        merged = df if merged is None else merged.merge(df, on="epoch", how="outer")
    if merged is None:
        raise ValueError("No logs were parsed successfully.")
    return merged.sort_values("epoch").reset_index(drop=True)


def plot_losses(merged: pd.DataFrame, log_names: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in log_names:
        ax.plot(
            merged["epoch"],
            merged[f"log10_total_{name}"],
            label=name,
            linewidth=1.2,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("log10 total loss")
    ax.set_title("Loss per epoch comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("loss_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare per-epoch loss histories stored in loss.out files."
    )
    parser.add_argument(
        "--soft",
        type=Path,
        default=Path("my_models/loss.out"),
        help="Path to the soft-constraint loss log.",
    )
    parser.add_argument(
        "--hard",
        type=Path,
        default=Path("my_models/loss_hard.out"),
        help="Path to the hard-constraint loss log.",
    )
    args = parser.parse_args()

    logs_to_compare = {}
    for label, path in (("soft_manual", args.soft), ("hard_manual", args.hard)):
        if not path.exists():
            raise FileNotFoundError(f"Log file not found: {path}")
        logs_to_compare[label] = path

    merged = load_and_merge(logs_to_compare)
    print("Merged loss summary (last 5 rows):")
    print(merged.tail())

    plot_losses(merged, list(logs_to_compare.keys()))


if __name__ == "__main__":
    main()
