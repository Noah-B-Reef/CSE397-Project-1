#!/usr/bin/env python3
"""
Compare soft- and hard-constraint PINN transmission spectra against the pRT
reference solution.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from petitRADTRANS import physical_constants as nc
except ImportError:
    import petitRADTRANS.physical_constants as nc

r_jup_mean = nc.r_jup_mean


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare soft and hard PINN transmission spectra against pRT."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/test_data_pinns.h5"),
        help="HDF5 file containing the true transmission spectra and atmosphere parameters.",
    )
    parser.add_argument(
        "--soft-eval",
        type=Path,
        required=True,
        help="Evaluation HDF5 file produced by eval_model.py for the soft PINN.",
    )
    parser.add_argument(
        "--hard-eval",
        type=Path,
        required=True,
        help="Evaluation HDF5 file produced by eval_model_hard.py for the hard PINN.",
    )
    parser.add_argument(
        "--index",
        type=int,
        nargs='+',
        default=[0],
        help="Spectrum index/indices to plot (0-based). Can specify multiple indices.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pinn_vs_prt.png"),
        help="Destination path for the generated comparison plot.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Load data
    true_transm = pd.read_hdf(args.data, key='transm')
    atm_params = pd.read_hdf(args.data, key='params')
    
    soft_transm = pd.read_hdf(args.soft_eval, key='transm')
    hard_transm = pd.read_hdf(args.hard_eval, key='transm')
    
    # Compute errors
    soft_err = soft_transm - true_transm
    hard_err = hard_transm - true_transm
    
    # Compute relative errors
    signal_strength = true_transm.values.max(axis=1) - true_transm.values.min(axis=1)
    rel_errors_soft = soft_err.values / signal_strength[:, None]
    rel_errors_hard = hard_err.values / signal_strength[:, None]
    
    indices = args.index if isinstance(args.index, list) else [args.index]
    n_samples = len(indices)
    
    # Create plots for each index
    for idx_num, ind in enumerate(indices):
        # Validate index
        if ind < 0 or ind >= len(true_transm):
            print(f"Warning: Index {ind} out of range [0, {len(true_transm)-1}], skipping.")
            continue
        
        # Create plot
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, 
                               gridspec_kw={'height_ratios': [3, 1]}, dpi=150)
        
        ax[0].plot(true_transm.columns, true_transm.iloc[ind, :] / r_jup_mean, label='pRT')
        ax[0].plot(true_transm.columns, soft_transm.iloc[ind, :] / r_jup_mean, 
                   label='Soft PINN', linewidth=0.5)
        ax[0].plot(true_transm.columns, hard_transm.iloc[ind, :] / r_jup_mean, 
                   label='Hard PINN', linewidth=0.5)
        
        ax[0].set_xscale('log')
        ax[0].set_ylabel(r'Transit radius ($\rm R_{Jup}$)')
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        
        ax[1].plot(true_transm.columns, rel_errors_soft[ind, :] * 100, 
                   label='Soft PINN', linewidth=0.5)
        ax[1].plot(true_transm.columns, rel_errors_hard[ind, :] * 100, 
                   label='Hard PINN', linewidth=0.5)
        
        ax[1].set_xlabel('Wavelength (microns)')
        ax[1].set_ylabel(r'$\Delta$R (%)')
        ax[1].set_ylim((-8.5, 8.5))
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
        
        plt.subplots_adjust(hspace=0.0)
        
        # Generate output filename
        if n_samples == 1:
            output_file = args.output
        else:
            output_stem = args.output.stem
            output_suffix = args.output.suffix
            output_file = args.output.parent / f"{output_stem}_idx{ind}{output_suffix}"
        
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
        plt.close()
        
        # Print metrics
        print(f"\n{'='*60}")
        print(f"Sample {ind}:")
        print(f"{'='*60}")
        print("\nAtmospheric parameters:")
        print(atm_params.iloc[ind])
        
        soft_rmse = np.sqrt(np.mean((rel_errors_soft[ind, :] * 100)**2))
        hard_rmse = np.sqrt(np.mean((rel_errors_hard[ind, :] * 100)**2))
        
        print(f"\nRMS relative error (%):")
        print(f"  Soft PINN: {soft_rmse:.3f}%")
        print(f"  Hard PINN: {hard_rmse:.3f}%")


if __name__ == "__main__":
    main()

