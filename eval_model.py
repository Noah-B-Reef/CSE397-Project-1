import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import inspect
try:
    from petitRADTRANS import Radtrans
except ImportError:
    from petitRADTRANS.radtrans import Radtrans

try:
    from petitRADTRANS import physical_constants as nc
except ImportError:
    import petitRADTRANS.physical_constants as nc
from tqdm import tqdm
import sys
import os

torch.set_num_threads(4)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if len(sys.argv) == 2:
    filename = sys.argv[-1]
elif len(sys.argv) == 3:
    filename = sys.argv[-1]
    model_file = sys.argv[-2]
else:
    raise ValueError("Please provide a filename as an argument ($ python eval_model.py path/example_file_name.h5")

true_transm = pd.read_hdf('data/test_data_pinns.h5', key='transm')
atm_params = pd.read_hdf('data/test_data_pinns.h5', key='params')

radtrans_kwargs = dict(
    line_species=['H2O'],
    rayleigh_species=['H2', 'He'],
    continuum_opacities=['H2-H2', 'H2-He'],
    wlen_bords_micron=[0.3, 15],
    enable_pinn=True,
)

accepted = inspect.signature(Radtrans.__init__).parameters
filtered_kwargs = {}
for key, value in radtrans_kwargs.items():
    if key in accepted:
        filtered_kwargs[key] = value
    else:
        print(f"Warning: Radtrans.__init__ does not accept argument '{key}'; skipping.")

atmosphere = Radtrans(**filtered_kwargs)

if len(sys.argv) == 3:
    # Set device for model loading
    if torch.cuda.is_available():
        dev = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        dev = torch.device('mps')
    else:
        dev = torch.device('cpu')
    
    atmosphere.dev = dev
    atmosphere.model = torch.load(os.path.join(model_file, 'TrainedModel/model.pkl'), map_location=dev, weights_only=False)

#genData = False
#if genData:
#    msa = torch.tensor(np.array([np.linspace(-18.0,-6.0,100), np.linspace(6/2,10/2,100)]).swapaxes(0,1))
#    std_rad = (np.sqrt(49.5**2 - (np.arange(100)-49.5)**2) / 49.5 * 0.05)
#    std_rad[0] = 1.0
#    std_rad[-1] = 1.0
#    msr = torch.tensor(np.array([np.linspace(1.0,-1.0,100),std_rad]).swapaxes(0,1))
#    # print(msa.shape, msr.shape)
#    atmosphere.mean_std_alpha = msa.to(atmosphere.dev)
#    atmosphere.mean_std_radii = msr.to(atmosphere.dev)

pressures = np.logspace(-6, 2, 100)
atmosphere.setup_opa_structure(pressures)

MMWs = {}
MMWs['H2'] = 2 * 1.008
MMWs['He'] = 4.0026
MMWs['H2O'] = 2 * 1.008 + 15.999
MMWs['CO'] = 12.011 + 15.999
MMWs['CO2'] = 12.011 + 2 * 15.999
MMWs['CH4'] = 12.011 + 4 * 1.008

transm_radii = []
errors = []

for i in tqdm(range(atm_params.shape[0])):
    
    R_pl = atm_params['R_pl'][i]*nc.r_jup_mean
    log_g = atm_params['log(g)'][i] # (SI units)
    gravity = 10**log_g * 100.0 # (SI to cgs!)
    P0 = 0.01 # pressure at R_pl

    temperature = atm_params['T'][i] * np.ones_like(pressures)

    vol_fractions = {}
    vol_fractions['H2O'] = (10.0**atm_params['log(H2O)'][i])
    vol_fractions['CO'] = (10.0**atm_params['log(CO)'][i])
    vol_fractions['CO2'] = (10.0**atm_params['log(CO2)'][i])
    vol_fractions['CH4'] = (10.0**atm_params['log(CH4)'][i])

    rest_vol = 1.0
    for vol_frac in vol_fractions.values():
        rest_vol -=  vol_frac
    
    vol_fractions['He'] = rest_vol * atm_params['He/H2'][i]
    vol_fractions['H2'] = rest_vol * (1.0 - atm_params['He/H2'][i])

    MMW = 0.0 * np.ones_like(pressures)
    for name, frac in vol_fractions.items():
        MMW += frac * MMWs[name]

    mass_fractions = {}
    for name, frac in vol_fractions.items():
        mass_fractions[name] = frac * MMWs[name] / MMW

    torch.set_num_threads(4)
    atmosphere.pressures = pressures
    # calculate transmission spectrum
    transmission, _, _ = atmosphere.calculate_transit_radii(temperature, mass_fractions, MMW, gravity, P0, R_pl)
    # transmission = np.swapaxes(transmission, 0, 1)
    err = transmission - true_transm.iloc[i,:]
    
    transm_radii.append(transmission)
    errors.append(err)


transm_radii = pd.DataFrame(transm_radii, columns=true_transm.columns)
errors = pd.DataFrame(errors, columns=true_transm.columns)

transm_radii.to_hdf(filename, key='transm', format='fixed')
errors.to_hdf(filename, key='error', format='fixed')
