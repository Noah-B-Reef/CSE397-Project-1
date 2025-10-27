#!/usr/bin/env python3
"""
Compare boundary-condition accuracy between two trained PINNs (soft vs. hard)
stored in the my_models directory.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import torch

from ImportFile import *  # noqa: F401,F403 - ensures Ec and helpers are initialised
from DatasetTorch2 import DefineDataset


def read_counts(info_path: Path) -> Dict[str, int]:
    with info_path.open("r") as handle:
        headers = handle.readline().strip().split(",")
        values = handle.readline().strip().split(",")
    raw = dict(zip(headers, values))
    return {
        "N_coll": int(float(raw["Nf_train"])),
        "N_u": int(float(raw["Nu_train"])),
        "N_int": int(float(raw["Nint_train"])),
    }


def read_batch_size(info_csv: Path) -> int:
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
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")
    load_kwargs = {"map_location": device}
    try:
        model = torch.load(model_path, weights_only=False, **load_kwargs)
    except TypeError:
        model = torch.load(model_path, **load_kwargs)
    model.eval()
    return model.to(device)


def boundary_error_metrics(
    model: torch.nn.Module,
    dataset: DefineDataset,
    device: torch.device,
) -> Tuple[float, float]:
    loader = dataset.data_boundary
    if len(loader) == 0:
        raise RuntimeError("Dataset does not contain boundary samples.")

    sq_errors = []
    abs_errors = []
    for x_b, u_b in loader:
        x_b = x_b.to(device)
        u_b = u_b.to(device)
        pred, target = Ec.apply_BC(x_b, u_b, model)
        diff = (pred - target).detach().cpu()
        sq_errors.append(diff.pow(2))
        abs_errors.append(diff.abs())

    sq = torch.cat(sq_errors)
    ab = torch.cat(abs_errors)
    rmse = float(torch.sqrt(sq.mean()))
    mae = float(ab.mean())
    return rmse, mae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare boundary-condition errors between soft and hard PINNs."
    )
    parser.add_argument(
        "--soft-model",
        type=Path,
        default=Path("my_models") / "RayleighOnly_lxlynorm_lowBC01_4+2x64_400x50_2",
        help="Directory containing InfoModel.txt and TrainedModel for the soft PINN.",
    )
    parser.add_argument(
        "--hard-model",
        type=Path,
        default=Path("my_models") / "RayleighOnly_hardBC_4+2x64_400x50_2",
        help="Directory containing InfoModel.txt and TrainedModel for the hard PINN.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when rebuilding the dataset batches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    soft_info = args.soft_model / "InfoModel.txt"
    hard_info = args.hard_model / "InfoModel.txt"
    soft_csv = args.soft_model / "TrainedModel" / "Information.csv"
    hard_csv = args.hard_model / "TrainedModel" / "Information.csv"

    for path in (soft_info, hard_info, soft_csv, hard_csv):
        if not path.exists():
            raise FileNotFoundError(f"Required metadata missing: {path}")

    soft_counts = read_counts(soft_info)
    hard_counts = read_counts(hard_info)
    if soft_counts != hard_counts:
        raise ValueError("Training datasets differ between the two models.")

    batch_size = read_batch_size(soft_csv)
    dataset = build_dataset(soft_counts, batch_size, seed=args.seed)

    device = Ec.dev
    soft_model = load_model(args.soft_model, device)
    hard_model = load_model(args.hard_model, device)

    soft_rmse, soft_mae = boundary_error_metrics(soft_model, dataset, device)
    hard_rmse, hard_mae = boundary_error_metrics(hard_model, dataset, device)

    print("Boundary condition error comparison (units: model output):")
    print(f"  Soft PINN -> RMSE: {soft_rmse:.6e}, MAE: {soft_mae:.6e}")
    print(f"  Hard PINN -> RMSE: {hard_rmse:.6e}, MAE: {hard_mae:.6e}")
    if hard_rmse < soft_rmse:
        print(f"  Hard PINN improves RMSE by {(soft_rmse - hard_rmse):.6e}")
    else:
        print(f"  Hard PINN RMSE is worse by {(hard_rmse - soft_rmse):.6e}")


if __name__ == "__main__":
    main()

