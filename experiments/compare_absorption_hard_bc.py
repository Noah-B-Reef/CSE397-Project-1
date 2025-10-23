#!/usr/bin/env python3
"""
Compare absorption-channel predictions of the scattering PINN with and without
hard boundary conditions. The script evaluates boundary accuracy, interior
residuals, and field-level deviations between the two trained models.
"""

from __future__ import annotations

import argparse
import csv
import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, Optional, Tuple

import pickle

import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ImportFile import *  # noqa: F401,F403 - ensures Ec and supporting modules are loaded

DEFAULT_STANDARD_DIR = Path("models") / "RayleighOnly_lxlynorm_lowBC01_4+2x64_400x50_2"
DEFAULT_HARD_DIR = Path("models") / "RayleighOnly_hardBC_4+2x64_400x50_2"


def load_network_properties(csv_path: Path) -> Dict[str, int]:
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
    return {
        "hidden_layers": int(row["hidden_layers"]),
        "neurons": int(row["neurons"]),
        "residual_parameter": float(row["residual_parameter"]),
        "kernel_regularizer": int(row["kernel_regularizer"]),
        "regularization_parameter": float(row["regularization_parameter"]),
        "batch_size": int(row["batch_size"]),
        "epochs": int(row["epochs"]),
        "activation": row["activation"],
    }


def load_dataset_counts(info_path: Path) -> Dict[str, int]:
    with open(info_path, "r") as handle:
        headers = handle.readline().strip().split(",")
        values = handle.readline().strip().split(",")
    raw = dict(zip(headers, values))
    return {
        "N_coll": int(float(raw["Nf_train"])),
        "N_u": int(float(raw["Nu_train"])),
        "N_int": int(float(raw["Nint_train"])),
    }


def build_dataset(
    counts: Dict[str, int],
    batch_size: int,
    seed: int = 42,
) -> DefineDataset:
    space_dims = Ec.space_dimensions
    time_dims = Ec.time_dimensions
    if time_dims == 0:
        n_initial = 0
        n_boundary = int(counts["N_u"] / (2 * space_dims)) if space_dims else 0
    else:
        n_boundary = int(counts["N_u"] / (4 * space_dims))
        n_initial = counts["N_u"] - 2 * space_dims * n_boundary
    dataset = DefineDataset(
        Ec.extrema_values,
        Ec.parameters_values if hasattr(Ec, "parameters_values") else None,
        Ec.type_of_points_dom if hasattr(Ec, "type_of_points_dom") else "uniform",
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
    try:
        model = torch.load(model_path, map_location=device)
    except pickle.UnpicklingError:
        model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model.to(device)


def normalize_inputs(x: torch.Tensor) -> torch.Tensor:
    x_coord = x[:, 0:1]
    y_coord = x[:, 1:2]
    phi = x[:, 2:3]
    r_pl = x[:, 3:4]
    a = x[:, 4:5]
    rayleigh = x[:, 5:6]
    delta_star = x[:, 6:7]

    r_pl_norm = ((r_pl / Ec.r_jup_mean) - 1.05) / 0.95
    a_norm = (torch.log10(a) - 7.1) / 0.35
    delta_norm = (delta_star - (0.5 * Ec.Delta_star_max / Ec.pi)) / (0.5 * Ec.Delta_star_max / Ec.pi)

    return torch.cat([x_coord, y_coord, phi, r_pl_norm, a_norm, rayleigh, delta_norm], dim=-1)


def absorption_boundary_metrics(
    model: torch.nn.Module,
    dataset: DefineDataset,
    device: torch.device,
) -> Optional[Dict[str, float]]:
    loader = dataset.data_boundary
    if len(loader) == 0:
        return None

    preds = []
    targets = []
    for x_b, u_b in loader:
        x_b = x_b.to(device)
        u_b = u_b.to(device)
        pred_flat, target_flat = Ec.apply_BC(x_b, u_b, model)
        pred = pred_flat.view(-1, 2)[:, 0].detach().cpu()
        target = target_flat.view(-1, 2)[:, 0].detach().cpu()
        preds.append(pred)
        targets.append(target)

    pred = torch.cat(preds)
    target = torch.cat(targets)
    error = pred - target
    return {
        "rmse": float(torch.sqrt((error ** 2).mean())),
        "mae": float(error.abs().mean()),
        "max_abs": float(error.abs().max()),
    }


def _compute_residual_quiet(
    model: torch.nn.Module,
    x_coll: torch.Tensor,
) -> torch.Tensor:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        res = Ec.compute_res(model, x_coll, Ec.space_dimensions, getattr(Ec, "obj", None), computing_error=True)
    return res


def absorption_residual_metrics(
    model: torch.nn.Module,
    dataset: DefineDataset,
    device: torch.device,
    max_points: int,
) -> Optional[Dict[str, float]]:
    loader = dataset.data_coll
    if len(loader) == 0 or max_points <= 0:
        return None

    collected = []
    total = 0
    for x_coll, _ in loader:
        x_coll = x_coll.to(device).detach().clone().requires_grad_(True)
        res = _compute_residual_quiet(model, x_coll)
        res_abs = res[:, 0].detach().cpu()
        collected.append(res_abs)
        total += res_abs.numel()
        if total >= max_points:
            break

    if not collected:
        return None

    residuals = torch.cat(collected)[:max_points]
    abs_res = residuals.abs()
    return {
        "rmse": float(torch.sqrt((residuals ** 2).mean())),
        "mean_abs": float(abs_res.mean()),
        "p95_abs": float(torch.quantile(abs_res, 0.95)),
        "max_abs": float(abs_res.max()),
    }


def absorption_field_difference(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    dataset: DefineDataset,
    device: torch.device,
    max_points: int,
) -> Optional[Dict[str, float]]:
    loader = dataset.data_coll
    if len(loader) == 0 or max_points <= 0:
        return None

    diffs = []
    total = 0
    for x_coll, _ in loader:
        x_coll = x_coll.to(device)
        inputs = normalize_inputs(x_coll)
        with torch.no_grad():
            u_a = model_a(inputs)[:, 0]
            u_b = model_b(inputs)[:, 0]
        diff = (u_a - u_b).detach().cpu()
        diffs.append(diff)
        total += diff.numel()
        if total >= max_points:
            break

    if not diffs:
        return None

    delta = torch.cat(diffs)[:max_points]
    abs_delta = delta.abs()
    return {
        "rmse": float(torch.sqrt((delta ** 2).mean())),
        "mean_abs": float(abs_delta.mean()),
        "p95_abs": float(torch.quantile(abs_delta, 0.95)),
        "max_abs": float(abs_delta.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--standard-dir",
        type=Path,
        default=DEFAULT_STANDARD_DIR,
        help="Directory containing the standard PINN (without hard BC).",
    )
    parser.add_argument(
        "--hard-dir",
        type=Path,
        default=DEFAULT_HARD_DIR,
        help="Directory containing the hard-BC PINN.",
    )
    parser.add_argument(
        "--collation-samples",
        type=int,
        default=20000,
        help="Maximum number of collocation points to include in residual and field metrics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when reconstructing the training dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    standard_csv = args.standard_dir / "TrainedModel" / "Information.csv"
    hard_csv = args.hard_dir / "TrainedModel" / "Information.csv"
    standard_info = args.standard_dir / "InfoModel.txt"
    hard_info = args.hard_dir / "InfoModel.txt"

    if not (standard_csv.exists() and standard_info.exists()):
        raise FileNotFoundError(f"Standard model metadata not found in {args.standard_dir}")
    if not (hard_csv.exists() and hard_info.exists()):
        raise FileNotFoundError(f"Hard-BC model metadata not found in {args.hard_dir}")

    standard_props = load_network_properties(standard_csv)
    standard_counts = load_dataset_counts(standard_info)
    hard_counts = load_dataset_counts(hard_info)
    if standard_counts != hard_counts:
        raise RuntimeError("Training sample counts differ between the two models; cannot build a common dataset.")

    device = Ec.dev if isinstance(Ec.dev, torch.device) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    dataset = build_dataset(standard_counts, standard_props["batch_size"], seed=args.seed)

    standard = load_model(args.standard_dir, device)
    hard = load_model(args.hard_dir, device)

    boundary_standard = absorption_boundary_metrics(standard, dataset, device)
    boundary_hard = absorption_boundary_metrics(hard, dataset, device)

    residual_standard = absorption_residual_metrics(standard, dataset, device, args.collation_samples)
    residual_hard = absorption_residual_metrics(hard, dataset, device, args.collation_samples)

    field_diff = absorption_field_difference(standard, hard, dataset, device, args.collation_samples)

    print("Absorption Channel Boundary Error (RMSE / MAE / max)")
    if boundary_standard is not None and boundary_hard is not None:
        print(
            f"  Standard : RMSE={boundary_standard['rmse']:.3e}, "
            f"MAE={boundary_standard['mae']:.3e}, "
            f"max={boundary_standard['max_abs']:.3e}"
        )
        print(
            f"  Hard BC  : RMSE={boundary_hard['rmse']:.3e}, "
            f"MAE={boundary_hard['mae']:.3e}, "
            f"max={boundary_hard['max_abs']:.3e}"
        )
    else:
        print("  Dataset does not contain boundary samples.")

    print("\nAbsorption Residual Statistics (RMSE / mean |res| / p95 |res| / max |res|)")
    if residual_standard is not None and residual_hard is not None:
        print(
            f"  Standard : RMSE={residual_standard['rmse']:.3e}, "
            f"mean|res|={residual_standard['mean_abs']:.3e}, "
            f"p95|res|={residual_standard['p95_abs']:.3e}, "
            f"max|res|={residual_standard['max_abs']:.3e}"
        )
        print(
            f"  Hard BC  : RMSE={residual_hard['rmse']:.3e}, "
            f"mean|res|={residual_hard['mean_abs']:.3e}, "
            f"p95|res|={residual_hard['p95_abs']:.3e}, "
            f"max|res|={residual_hard['max_abs']:.3e}"
        )
    else:
        print("  Collocation loader is empty; residual statistics unavailable.")

    print("\nAbsorption Field Difference (standard - hard)")
    if field_diff is not None:
        print(
            f"  RMSE={field_diff['rmse']:.3e}, "
            f"mean|Δ|={field_diff['mean_abs']:.3e}, "
            f"p95|Δ|={field_diff['p95_abs']:.3e}, "
            f"max|Δ|={field_diff['max_abs']:.3e}"
        )
    else:
        print("  Unable to evaluate field-level differences.")


if __name__ == "__main__":
    main()
