import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from ImportFile import *  # noqa: F401,F403 pylint: disable=wildcard-import,unused-wildcard-import
from DatasetTorch2 import DefineDataset

# Ensure PyTorch allows loading the serialized model classes
SAFE_GLOBALS = []
for cls_name in ("Pinns", "PinnsHardBC", "PinnsRes", "ResidualBlock"):
    cls = globals().get(cls_name)
    if cls is not None:
        SAFE_GLOBALS.append(cls)
if SAFE_GLOBALS:
    try:
        torch.serialization.add_safe_globals(SAFE_GLOBALS)
    except AttributeError:
        # Older torch versions do not expose add_safe_globals.
        pass


def read_training_counts(info_path: Path) -> Dict[str, int]:
    with info_path.open("r") as handle:
        headers = handle.readline().strip().split(",")
        values = handle.readline().strip().split(",")
    data = dict(zip(headers, values))
    return {
        "N_coll": int(float(data["Nf_train"])),
        "N_u": int(float(data["Nu_train"])),
        "N_int": int(float(data["Nint_train"])),
    }


def read_batch_size(info_csv: Path) -> int:
    import csv

    with info_csv.open("r", newline="") as handle:
        row = next(csv.DictReader(handle))
    return int(row["batch_size"])


def build_dataset(counts: Dict[str, int], batch_size: int, seed: int = 42) -> DefineDataset:
    space_dims = Ec.space_dimensions
    time_dims = Ec.time_dimensions

    if space_dims == 0:
        n_boundary = 0
    elif time_dims == 0:
        n_boundary = counts["N_u"] // (2 * space_dims)
    else:
        n_boundary = counts["N_u"] // (4 * space_dims)

    if time_dims == 1:
        n_initial = counts["N_u"] - 2 * space_dims * n_boundary
    else:
        n_initial = 0

    dataset = DefineDataset(
        Ec.extrema_values,
        getattr(Ec, "parameters_values", None),
        getattr(Ec, "type_of_points_dom", "uniform"),
        counts["N_coll"],
        n_boundary,
        n_initial,
        counts["N_int"],
        batches=batch_size,
        output_dimension=Ec.output_dimension,
        space_dimensions=space_dims,
        time_dimensions=time_dims,
        parameter_dimensions=getattr(Ec, "parameter_dimensions", 0),
        random_seed=seed,
        obj=getattr(Ec, "obj", None),
        shuffle=False,
        type_point_param=getattr(Ec, "type_of_points", None),
    )
    dataset.assemble_dataset()
    return dataset


def load_model(model_dir: Path, device: torch.device) -> torch.nn.Module:
    model_path = model_dir / "TrainedModel" / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model checkpoint: {model_path}")
    load_kwargs = {"map_location": device}
    try:
        model = torch.load(model_path, weights_only=False, **load_kwargs)
    except TypeError:
        # Older PyTorch versions do not support weights_only.
        model = torch.load(model_path, **load_kwargs)
    model.eval()
    return model.to(device)


def gather_supervised(dataset: DefineDataset) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for xb, ub in dataset.data_boundary:
        xs.append(xb)
        ys.append(ub)
    for xu, u in dataset.data_initial_internal:
        xs.append(xu)
        ys.append(u)
    if not xs:
        raise RuntimeError("Dataset does not contain supervised samples.")
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().reshape(-1)


def compute_predictions(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        return model(inputs.to(device)).cpu()


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def prepare_plot_data(
    true_vals: np.ndarray,
    pred_soft: np.ndarray,
    pred_hard: np.ndarray,
    sample: Optional[int] = 8000,
) -> Dict[str, np.ndarray]:
    if sample is not None and true_vals.size > sample:
        rng = np.random.default_rng(0)
        idx = rng.choice(true_vals.size, size=sample, replace=False)
    else:
        idx = slice(None)

    return {
        "u_true_plot": true_vals[idx],
        "u_soft_plot": pred_soft[idx],
        "u_hard_plot": pred_hard[idx],
    }


def plot_comparison(
    plot_data: Dict[str, np.ndarray],
    errors_soft: np.ndarray,
    errors_hard: np.ndarray,
    rmse_soft: float,
    rmse_hard: float,
    output_path: Path,
) -> None:
    u_true_plot = plot_data["u_true_plot"]
    u_soft_plot = plot_data["u_soft_plot"]
    u_hard_plot = plot_data["u_hard_plot"]

    all_vals = np.concatenate([u_true_plot, u_soft_plot, u_hard_plot])
    lo, hi = all_vals.min(), all_vals.max()
    pad = 0.02 * (hi - lo if hi > lo else 1.0)
    limits = (lo - pad, hi + pad)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].scatter(u_true_plot, u_soft_plot, s=10, alpha=0.3)
    axes[0].plot(limits, limits, "--", color="k", linewidth=1)
    axes[0].set_xlim(limits)
    axes[0].set_ylim(limits)
    axes[0].set_aspect("equal", "box")
    axes[0].set_title(f"Soft constraints (RMSE={rmse_soft:.2e})")
    axes[0].set_xlabel("True u")
    axes[0].set_ylabel("Predicted u")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(u_true_plot, u_hard_plot, s=10, alpha=0.3, color="tab:orange")
    axes[1].plot(limits, limits, "--", color="k", linewidth=1)
    axes[1].set_xlim(limits)
    axes[1].set_ylim(limits)
    axes[1].set_aspect("equal", "box")
    axes[1].set_title(f"Hard constraints (RMSE={rmse_hard:.2e})")
    axes[1].set_xlabel("True u")
    axes[1].set_ylabel("Predicted u")
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(
        errors_soft,
        bins=60,
        alpha=0.6,
        density=True,
        label="Soft constraints",
    )
    axes[2].hist(
        errors_hard,
        bins=60,
        alpha=0.6,
        density=True,
        label="Hard constraints",
    )
    axes[2].set_xlabel("|u_pred - u_true|")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Absolute error distribution")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle("Model comparison vs. true solution", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(output_path, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two trained models against the true solution."
    )
    parser.add_argument(
        "--soft-model",
        type=Path,
        required=True,
        help="Directory containing InfoModel.txt and TrainedModel for the soft-constraint model.",
    )
    parser.add_argument(
        "--hard-model",
        type=Path,
        required=True,
        help="Directory containing InfoModel.txt and TrainedModel for the hard-constraint model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_error_comparison.png"),
        help="Path to save the comparison plot.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=8000,
        help="Number of samples to plot (use all samples if smaller).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset sampling and plotting.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    soft_info = args.soft_model / "InfoModel.txt"
    hard_info = args.hard_model / "InfoModel.txt"
    soft_csv = args.soft_model / "TrainedModel" / "Information.csv"
    hard_csv = args.hard_model / "TrainedModel" / "Information.csv"

    for path in (soft_info, hard_info, soft_csv, hard_csv):
        if not path.exists():
            raise FileNotFoundError(f"Required metadata missing: {path}")

    soft_counts = read_training_counts(soft_info)
    hard_counts = read_training_counts(hard_info)
    if soft_counts != hard_counts:
        raise ValueError("Training datasets differ between the two models.")

    batch_size = read_batch_size(soft_csv)
    dataset = build_dataset(soft_counts, batch_size, seed=args.seed)

    device = Ec.dev
    soft_model = load_model(args.soft_model, device)
    hard_model = load_model(args.hard_model, device)

    x_sup, u_true = gather_supervised(dataset)
    u_true_np = to_numpy(u_true)

    soft_pred = compute_predictions(soft_model, x_sup, device)
    hard_pred = compute_predictions(hard_model, x_sup, device)

    soft_np = to_numpy(soft_pred)
    hard_np = to_numpy(hard_pred)

    errors_soft = np.abs(soft_np - u_true_np)
    errors_hard = np.abs(hard_np - u_true_np)

    rmse_soft = rmse(soft_np, u_true_np)
    rmse_hard = rmse(hard_np, u_true_np)

    print(f"Soft model RMSE: {rmse_soft:.6e}")
    print(f"Hard model RMSE: {rmse_hard:.6e}")
    print(f"Mean absolute error soft: {errors_soft.mean():.6e}")
    print(f"Mean absolute error hard: {errors_hard.mean():.6e}")

    plot_data = prepare_plot_data(
        u_true_np, soft_np, hard_np, sample=args.sample
    )
    plot_comparison(
        plot_data,
        errors_soft,
        errors_hard,
        rmse_soft,
        rmse_hard,
        args.output,
    )
    print(f"Saved comparison plot to {args.output}")


if __name__ == "__main__":
    main()
