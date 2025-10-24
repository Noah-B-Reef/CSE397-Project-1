# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a PyTorch-based implementation of Physics-Informed Neural Networks (PINNs) for approximating atmospheric scattering in exoplanet atmospheres. The project is based on [Dahlbüdding et al. (2024)](https://arxiv.org/abs/2408.00084) and implements two main PINN variants:
1. **Absorption PINN** - Models atmospheric absorption
2. **Rayleigh Scattering PINN** - Models Rayleigh scattering with special architecture

## Common Commands

### Training

Train a PINN with default parameters:
```bash
python3 PINNS2.py
```

Train with specific random seed:
```bash
python3 PINNS2.py <seed_number>
```

Train with hard boundary conditions (alternative architecture):
```bash
python3 PINNS2_hard.py
```

Hyperparameter search (runs training with seeds 1-7):
```bash
bash hyperparameter.sh
```

### Evaluation

Evaluate a trained model on test spectra:
```bash
python3 eval_model.py path/to/model.pkl path/to/output_file.h5
```

Evaluate with hard BC model:
```bash
python3 eval_model_hard.py path/to/model.pkl path/to/output_file.h5
```

Run evaluation for hyperparameter search:
```bash
bash hyperparameter_eval.sh
```

### Jupyter Notebooks

Start Jupyter to use the analysis notebooks:
```bash
jupyter notebook JupyterNotebooks/
```

## Architecture and Key Components

### Core Training Files

- **PINNS2.py / PINNS2_hard.py**: Main training scripts that initialize networks, datasets, and run the optimization loop. Configure network properties at the top of these files before training.
- **ImportFile.py**: Central imports file that specifies which equation model to use. Change the import statement here to switch between absorption and scattering models.

### Model Architecture

- **ModelClassTorch2.py**: Defines neural network architectures
  - `Pinns`: Base PINN class with optional mask for scattering (set `addmask=True/False` in line 60)
  - `PinnsHardBC`: PINN variant that enforces boundary conditions exactly in the forward pass
  - `PinnsRes`: Residual block-based architecture
  - Custom loss function `CustomLoss` that combines physics residuals and boundary/initial conditions

### Equation Models (EquationModels/)

Switch between these by modifying the import in `ImportFile.py`:

- **IsothermalAtmo.py**: Absorption-only model
  ```python
  import EquationModels.IsothermalAtmo as Ec
  ```
  Set `addmask=False` in ModelClassTorch2.py

- **IsothermalAtmoOnlyRayleigh.py**: Rayleigh scattering model  
  ```python
  import EquationModels.IsothermalAtmoOnlyRayleigh as Ec
  ```
  Set `addmask=True` in ModelClassTorch2.py

- **IsothermalAtmoRayleigh.py**: Combined model

Each equation model defines:
- Domain boundaries and dimensions (x, y, φ coordinates)
- Physics equations (radiative transfer equation)
- Boundary conditions via `list_of_BC`
- Loss computation including physics residuals

### Dataset Generation

- **DatasetTorch2.py**: `DefineDataset` class handles sampling of collocation points, boundary points, and initial conditions. Points can be sampled uniformly or using Latin Hypercube/Sobol sequences.

### Object Classes

- **ObjectClass.py**: Defines geometric objects (Cylinder, Square) for handling solid boundaries in flow problems. Not heavily used in this atmospheric scattering application.

## Network Configuration

Network properties are defined in the training scripts (PINNS2.py, PINNS2_hard.py) as a dictionary:

```python
network_properties = {
    "hidden_layers": 4,        # Number of hidden layers
    "neurons": 64,             # Neurons per layer
    "residual_parameter": 0.5, # Weight for residual loss
    "kernel_regularizer": 2,   # L2 regularization type
    "regularization_parameter": 0,
    "batch_size": 32768*4*4,   # Training batch size
    "epochs": 400,             # Number of epochs
    "activation": "tanh"       # Activation function
}
```

## Training Process

1. **Optimizer**: Two-stage training
   - ADAM optimizer for initial epochs (fast convergence)
   - L-BFGS optimizer for final refinement (higher accuracy)
   - Configured in PINNS2.py around lines 240-243

2. **Loss Function**: Physics-informed loss combining:
   - Residual of the radiative transfer equation (computed via automatic differentiation)
   - Boundary condition violations
   - Initial condition errors (if time-dependent)

3. **Device Support**: Automatically detects and uses:
   - CUDA (NVIDIA GPUs)
   - MPS (Apple Silicon)
   - CPU fallback

## Integration with petitRADTRANS

The trained PINNs can replace expensive opacity calculations in petitRADTRANS:

1. Replace `site-packages/petitRADTRANS/radtrans.py` with `pRT/radtrans.py` from this repo
2. Place trained `model.pkl` in pRT's `input_data/pinn/` (absorption) or `input_data/rayleigh_pinn/` (scattering) folders
3. Enable in atmosphere initialization:
   ```python
   atmosphere = Radtrans([...], enable_pinn=True, enable_rayleigh=True)
   ```

## Data

- **data/test_data_pinns.h5**: Test dataset with 95 atmospheric spectra
- Trained models available at: https://zenodo.org/doi/10.5281/zenodo.10727497

## Key Physics

- **Coordinates**: (x, y, φ) where x and y are normalized spatial coordinates in [-1, 1], and φ is the angular coordinate
- **Radiative Transfer**: Solves the equation for light intensity through scattering/absorbing atmospheres
- **Boundary Conditions**: Stellar illumination at atmosphere boundaries
- **Isothermal Atmosphere**: Assumes constant temperature with pressure-dependent radius profile

## Jupyter Notebooks

1. **1a_HPsearch_loss.ipynb**: Visualize loss curves from hyperparameter search
2. **1b_HPsearch_allErrors.ipynb**: Compare test spectrum errors across hyperparameters
3. **2a_absPINN_individualEvalData.ipynb**: Analyze absorption PINN on individual test cases
4. **2b_absPINN_alphaProfileAnalysis.ipynb**: Examine learned opacity profiles
5. **2c_absPINN_testRandInputs.ipynb**: Test PINN on randomly generated profiles
6. **3_scaPINN_plot_u+residual.ipynb**: Visualize scattering PINN solution and residuals
7. **4_scaPINN_testRayleigh.ipynb**: Test and validate Rayleigh scattering implementation
8. **5_CompareScatteringSolutions.ipynb**: Compare different scattering solution methods

## Model Outputs

Trained models are saved to `models/` with structure:
```
models/<model_name>/
├── TrainedModel/
│   ├── model.pkl          # PyTorch model
│   └── Information.csv    # Network hyperparameters
└── InfoModel.txt          # Training metadata (time, loss, etc.)
```

## Important Implementation Details

- **Automatic Differentiation**: Physics residuals computed via `torch.autograd` for gradient calculations with respect to input coordinates
- **Masking for Scattering**: The `addmask` parameter in `Pinns` class adds a specialized output channel for scattering that respects angular constraints
- **Random Seeding**: Control via `sampling_seed` parameter for reproducible dataset sampling
- **Batch Generation**: For scattering model, training data can be generated on-the-fly or pre-sampled (controlled by `generate_train_data` in equation models)
