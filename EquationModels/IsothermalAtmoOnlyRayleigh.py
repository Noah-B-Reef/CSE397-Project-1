from ImportFile import *
import time

pi = math.pi

extrema_values = None
space_dimensions = 3
time_dimensions = 0
domain_values = torch.tensor([[-1.0,1.0], [-1.0,1.0], [-1.0,1.0]]) # x,y,phi
parameters_values = None # parameters added using append_rand_int()

type_of_points = "uniform"
type_of_points_dom = "uniform"
input_dimensions = 7
output_dimension = 1

causality = False # No influence here, but has to be defined for ModelClassTorch2.py

generate_train_data = True # if training data is completely generated on the fly
# (otherwise the a large sample of radius parameters will be computed beforehand and a random subsample is used during training)
n_before = 1e7 # number of training points that are precomputed and sampled from during training

resGrad = False # if gradient of residual with respect to the inputs is added to the loss (Yu et al. 2021, Gradient-enhanced PINNs [...])
resGradFactor = 0.001

ub_0 = 1.0
#strip = 0.05
r_jup_mean = 6991100000.0

# if fixed_values != None, then train PINN with fixed params (only x,y,mu are not fixed)
fixed_values = None # torch.tensor([1.05*r_jup_mean, 10**7.1, 1.4, 0.1]).reshape(1,4)

n_quad_abs = 5  # number of integration points for absorption term (should be odd, such that zero is included)
n_quad_sca = 14 # number of integration points for scattering term

if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device("cpu")
    
res2_param = torch.tensor([1.0]).to(dev) # weight of the residual of the complete RTE w.r.t. the "only-absorption" RTE
resDiffBC_param = torch.tensor([0.1]).to(dev) # weight of the BC at the lower boundary (y = -1)
#resDiffpos_param = torch.tensor([0.1]).to(dev)

mean_rayScat = torch.linspace(-16.7, -8.7, 100, dtype=torch.float32)
mean_rayScat_gpu = torch.linspace(-16.7, -8.7, 100, dtype=torch.float32).to(dev)
std_rayScat = 3.0
Delta_star_max = 0.3*pi
# references: R* = 1 R_sun, a = 0.01 AU -> Delta_star_max = 0.15 pi
# OR Wasp-39b: R* = 0.92 R_sun, a = 0.05 AU -> Delta_star_max = 0.03 pi

def two_value_interpolation(x, y, val):
    """
    Linearly interpolates for a target value `val` based on tensors `x` and `y`.

    Args:
    - x (Tensor): Input values.
    - y (Tensor): Corresponding values to `x`.
    - val (float): Target value for interpolation.

    Returns:
    - Tensor: Interpolated values at `val`.

    Handles out-of-range `val` by adjusting to nearest values in `x` and `y`.
    """
    index = (x > val).sum(dim=1, keepdim=True)
    outside_atm = (index == 0)
    index[outside_atm] += 1
    _xrange = x.gather(1, index) - x.gather(1, index - 1)
    xdiff = val - x.gather(1, index - 1)
    modolo = xdiff / _xrange
    ydiff = y.gather(1, index) - y.gather(1, index - 1)
    interpolated_y = y.gather(1, index - 1) + modolo * ydiff
    interpolated_y[outside_atm] = y[outside_atm[:,0], 0] - 3.0
    return interpolated_y

def append_rand_int(rand_samples, dim, random_seed):
    """
    Generates training data by appending random alphas and radii to `rand_samples`
    based on a condition. If generating training data, it uses custom functions to generate
    alphas and radii. Otherwise, it selects from fixed arrays using a random index.

    Args:
    - rand_samples (Tensor): The input samples to which random alphas and radii will be appended.
    - dim (int): The dimension along which to append the random alphas and radii.
    - random_seed (int): The seed for random number generation to ensure reproducibility.

    Returns:
    - Tensor: The input tensor augmented with randomly generated alphas and radii.
    """
    n = rand_samples.shape[0]
    
    if generate_train_data:
        gen_radii = get_radii(n, random_seed=random_seed)
        gen_rayScat = get_rayScat(n, random_seed=random_seed)
    else:
        torch.manual_seed(random_seed)
        ind_radii = torch.randint(radii_fixed.shape[0], size=(n,))
        gen_radii = radii_fixed[ind_radii,:]
        gen_rayScat = get_rayScat(n, random_seed=random_seed)
        
    rand_samples = torch.cat([rand_samples,gen_radii,gen_rayScat], dim=1)
    
    return rand_samples

def radius(x,y,r0,r99):
    """
    Calculates the radius based on the given parameters.
    
    Args:
    - x, y (Tensor): Input tensors representing coordinates.
    - r0, r99 (float): Constants defining the radius profile r(P) at specific conditions.
    
    Returns:
    - Tensor: Calculated radius for each pair of x and y.
    """
    dx2 = (x**2) * (r0**2 - r99**2)
    dy2 = (((y+1) * (r0-r99)/2)+r99)**2
    return torch.sqrt(dx2 + dy2)

def r_isotherm(a, r_P0, gpu=True):
    """
    Calculates the isothermal radius profile r(P) for different pressures,
    where log_10(P[bar]) in [-6,2] with 100 layers
    
    Args:
    - a (Tensor): Scale factor.
    - r_P0 (float): Radius at a reference pressure P0.
    - gpu (bool, optional): If True, use GPU for calculations. Defaults to True.
    
    Returns:
    - Tensor: Calculated isothermal radius values.
    """
    P0 = 0.01
    if gpu:
        pressures = torch.logspace(-6, 2, 100).unsqueeze(0).to(dev)
    else:
        pressures = torch.logspace(-6, 2, 100).unsqueeze(0)
    return r_P0 / (1 + ((a/r_P0)*torch.log(pressures/P0)))
    
def scattering_phase_fct(mu, mu_prime):
    """
    Computes the scattering phase function for given angles.

    Args:
    - mu, mu_prime (Tensor): Cosines of the scattering angles.
    
    Returns:
    - Tensor: The phase function value.
    """
    phi = 0.75*(1 + (mu**2)*(mu_prime**2) + 0.5*(1-mu**2)*(1-mu_prime**2))
    return phi

def scattering_integral_abs_factor(phi, phi_prime, Delta_star):
    """
    Calculates the factor in the integrand of the scattering integral with respect to u_abs
    for a given scattering angle and angular extent of the star.

    Args:
    - phi, phi_prime (Tensor): Scattering angles.
    - Delta_star (Tensor): (Half the) angular extent of the star.
    
    Returns:
    - Tensor: Integrand factor for scattered u_abs.
    """
    sqroot = torch.sqrt( 1 - (torch.cos(Delta_star)**2/torch.cos(phi_prime)**2) )
    return - 1.5 * sqroot * ( torch.cos(phi-phi_prime)**2 * ((sqroot**2)/3 - 1) - 1 )

def scattering_integral_sca_factor(phi, theta_prime):
    """
    Calculates the factor in the integrand of the scattering integral with respect to u_sca
    for given scattering angles.

    Args:
    - phi, theta_prime (Tensor): Scattering angles.
    
    Returns:
    - Tensor: Integrand factor for scattered u_sca.
    """
    theta = torch.abs(phi)
    cos_sq = torch.cos(theta)**2 * torch.cos(theta_prime)**2
    sin_sq = torch.sin(theta)**2 * torch.sin(theta_prime)**2
    sin_2  = torch.sin(2*theta) * torch.sin(2*theta_prime)
    
    factor_neg = 0.75*( pi*(1 + cos_sq - 0.5*sin_sq) + torch.sign(phi) * sin_2 )
    factor_pos = 0.75*( pi*(1 + cos_sq + 0.5*sin_sq) - torch.sign(phi) * sin_2 )
    
    return (factor_neg, factor_pos)

def compute_scattering(x, model):
    """
    Computes scattering integral values based on input features and a model.

    Args:
    - x (Tensor): Input features with dimensions for scattering calculations.
    - model (Model): A PyTorch model for computing outputs based on `x`.
    
    Returns:
    - Tensor: Normalized scattering integral values.
    """
    
    Delta_star = x[:,6].unsqueeze(1) * (0.5*Delta_star_max/pi) + (0.5*Delta_star_max/pi)

    phi_prime = torch.linspace(0,1,int(n_quad_sca/2)).repeat(x.shape[0],1).to(dev) * (1 - Delta_star/pi) + Delta_star
    #phi_prime = phi_prime[torch.abs(phi_prime) > Delta_star]
    phi_prime = torch.cat([phi_prime, -phi_prime, Delta_star*(torch.linspace(-1,1,n_quad_abs).to(dev)).repeat(x.shape[0],1)], dim=-1)
    phi_prime, _ = torch.sort(phi_prime, dim=-1)
    phi_prime = phi_prime.unsqueeze(2)
    
    phi = x[:,2]
    #print(mu.shape, mu_prime.shape)
    
    x_l = x[:,:2].unsqueeze(1)
    x_l_rest = x[:,3:].unsqueeze(1)
    
    inputs = [torch.repeat_interleave(x_l, phi_prime.shape[1], dim=1), phi_prime, torch.repeat_interleave(x_l_rest, phi_prime.shape[1], dim=1)]
    inputs = torch.cat(inputs, dim=-1)
    #print(inputs.shape)
    inputs = inputs.reshape(-1,7)

    u = model(inputs)
    #print(u.shape) (n*n_quad,2)
    u = u.reshape((x.shape[0], phi_prime.shape[1], 2))
    #print(u.shape) (n,n_quad,2)
    
    Delta_star = Delta_star * pi
    phi = phi * pi
    phi_prime = phi_prime[:,:,0] * pi
    
    scatter_values = torch.zeros_like(phi).to(dev)
    
    # calculate scattering integral for absorption term
    abs_mask = torch.zeros(phi_prime.shape[1], dtype=bool).to(dev)
    abs_mask[int(n_quad_sca/2):int(n_quad_sca/2)+n_quad_abs] = True
    abs_factor = scattering_integral_abs_factor(phi.unsqueeze(1), phi_prime[:,abs_mask].detach(), Delta_star.detach())
    scatter_values += torch.trapezoid(abs_factor*u[:,abs_mask,0], x=phi_prime[:,abs_mask], dim=-1)
    
    # calculate scattering integral for scattering term
    phi_prime = phi_prime[:, phi_prime[0,:] >= 0] # convert phi_prime to theta_prime in [0,pi]
    sca_factor_neg, sca_factor_pos = scattering_integral_sca_factor(phi.unsqueeze(1), phi_prime)
    u_sca_neg, u_sca_pos = torch.tensor_split(u[:,:,1], 2, dim=1)
    u_sca_neg = torch.flip(u_sca_neg, [1]) # invert order of u_sca_neg [-pi,0] -> [0,-pi]
    u_sca_pos = torch.cat([u_sca_neg[:,:1], u_sca_pos], dim=-1) # append first element of u_sca_neg to first position of u_sca_pos (where phi_prime=0)
    sca_integrand = torch.abs(torch.sin(phi_prime)) * (sca_factor_neg*u_sca_neg + sca_factor_pos*u_sca_pos)
    scatter_values += torch.trapezoid(sca_integrand, x=phi_prime, dim=-1)
    
    return (scatter_values / (4*pi)) # normalize


def get_radii(n, random_seed=42):
    """
    Generates radii and scale height values for `n` samples.

    Args:
    - n (int): Number of samples.
    - random_seed (int): Seed for random number generation.
    
    Returns:
    - Tensor: Radii and semi-major axis values for the samples.
    
    The function initializes a tensor for radii, generates planet radii (`R_pl`) within a specific range
    scaled by the mean radius of Jupiter, and calculates scale height (`a`) values from a (log) normal distribution.
    """
    torch.manual_seed(random_seed)
    
    radii = torch.zeros((n, 2))
    
    R_pl = (torch.rand((n,)) * 1.8 + 0.2) * r_jup_mean
    a = 10**torch.normal(7.1,0.35, size=(n,))
    
    radii[:,0] = R_pl
    radii[:,1] = a
    
    return radii

if not generate_train_data:
    # pre-calculate large sample (n_before) of radius parameters for (somewhat) faster training
    torch.manual_seed(42)
    radii_fixed = get_radii(int(n_before))
    print('radii shape: ', radii_fixed.shape)
    real_radii_fixed = r_isotherm(radii_fixed[:,1].unsqueeze(1), radii_fixed[:,0].unsqueeze(1), gpu=False)[:,0]
    radii_fixed = radii_fixed[torch.logical_and(real_radii_fixed>0.0, real_radii_fixed<2.5*r_jup_mean),:]
    print('radii shape: ', radii_fixed.shape)
    
    
def get_rayScat(n, random_seed=42):
    """
    Generates Rayleigh scattering parameters for `n` samples.

    Args:
    - n (int): Number of samples.
    - random_seed (int): Seed for random number generation.
    
    Returns:
    - Tensor: Alpha Rayleigh scattering and mu_max parameters.
    """

    alpha_rayScat = torch.rand((n,1)) * 4.0 - 2.0
    # alpha_rayScat = alpha_rayScat * std_rayScat + mean_rayScat[-1]
    mu_max = torch.rand((n,1)) * Delta_star_max/pi
    
    return torch.cat([alpha_rayScat, mu_max], dim=-1)

def generator_samples(type_point_param, samples, dim, random_seed, extend=1.0):
    """
    Generates sample parameters based on specified domain, type, and optional constraints.
    
    Args:
    - type_point_param (str): Type of sampling point parameter (only "uniform" works).
    - samples (int): Number of samples to generate.
    - dim (int): Dimension of the parameter space.
    - random_seed (int): Seed for random number generation.
    - extend (float, optional): Extension factor for the first parameter dimension. Defaults to 1.0. [DEPRECATED]
    
    Returns:
    - Tensor: Generated sample parameters.
    """

    if parameters_values is None:
        extrema = domain_values
    else:
        extrema = torch.cat([domain_values, parameters_values], 0)
    extrema_0 = extrema[:, 0]
    extrema_f = extrema[:, 1]

    extrema_f[0] = extend * (extrema_f[0] - extrema_0[0]) + extrema_0[0]

    if type_point_param == "uniform":
        if random_seed is not None:
            torch.random.manual_seed(random_seed)
        params = torch.rand([samples, dim]).type(torch.FloatTensor) * (extrema_f - extrema_0) + extrema_0
        if fixed_values is not None:
            params = torch.cat([params,fixed_values.repeat(params.shape[0], 1)], dim=1)
            params[:,-1:] = torch.rand((params.shape[0],1)) * Delta_star_max/pi
        else:
            params = append_rand_int(params, dim, random_seed)
        return params


def compute_res(network, x_f_train, space_dimensions, solid_object, computing_error, sample=True):
    """
    Computes the residual of the network's predictions against physical laws (RTE).
    
    Args:
    - network (torch.nn.Module): The neural network model.
    - x_f_train (Tensor): Training data inputs.
    - space_dimensions (int): Number of spatial dimensions in the data.
    - solid_object: -
    - computing_error: -
    - sample (bool): If half of phi is sampled within [-Delta_*, +Delta_*]. Default: True
    
    Returns:
    - Tensor: The computed residuals.
    """
    x = x_f_train[:, 0].unsqueeze(1)
    y = x_f_train[:, 1].unsqueeze(1)
    phi = x_f_train[:, 2].unsqueeze(1)
    Delta_star = x_f_train[:,6].unsqueeze(1)
    if sample:
        # sample half of phi's within Delta_star range
        phi[::2] = phi[::2] * Delta_star[::2]
    Delta_star_norm = (Delta_star - (0.5*Delta_star_max/pi)) / (0.5*Delta_star_max/pi)
    
    r_pl = x_f_train[:, 3].unsqueeze(1)
    a = x_f_train[:, 4].unsqueeze(1)

    
    real_gen_radii = r_isotherm(a, r_pl)
    r_pl = ((r_pl / r_jup_mean) - 1.05) / 0.95
    a = (torch.log10(a) - 7.1) / 0.35

    r0 = real_gen_radii[:,0].unsqueeze(1)
    r99 = real_gen_radii[:,-1].unsqueeze(1)
    r = radius(x, y, r0, r99)

    l_x = r0 * torch.sqrt( 1 - (r99/r0)**2)
    l_y = 0.5*(r0-r99)

    #print(y.shape, l_x.shape, l_y.shape)
    #print(torch.min(l_y/l_x), torch.max(l_y/l_x))
    #y_norm = y * l_y/l_x

    rayScat_train = x_f_train[:,5].unsqueeze(1)

    x_f_train_norm = torch.cat([x, y, phi, r_pl, a, rayScat_train, Delta_star_norm], dim=-1).float()
    x_f_train_norm.requires_grad = True

    u = network(x_f_train_norm).reshape(-1,2)
    
    grad_u1 = torch.autograd.grad(u[:,0], x_f_train_norm, grad_outputs=torch.ones(x_f_train_norm.shape[0]).to(dev), create_graph=True)[0]
    
    grad_u1_x = grad_u1[:, 0].unsqueeze(1)
    grad_u1_y = grad_u1[:, 1].unsqueeze(1)
    
    grad_u2 = torch.autograd.grad((u[:,0]+u[:,1]), x_f_train_norm, grad_outputs=torch.ones(x_f_train_norm.shape[0]).to(dev), create_graph=True)[0]
    
    grad_u2_x = grad_u2[:, 0].unsqueeze(1)
    grad_u2_y = grad_u2[:, 1].unsqueeze(1)

    #grad_u2_x += grad_u1_x
    #grad_u2_y += grad_u1_y

    #r0 = real_gen_radii[:,0].unsqueeze(1)
    #r99 = real_gen_radii[:,-1].unsqueeze(1)
    #r = radius(x, y, r0, r99)
    
    #l_x = r0 * torch.sqrt( 1 - (r99/r0)**2)
    #l_y = 0.5*(r0-r99)
    
    real_rayScat = (rayScat_train * std_rayScat) + mean_rayScat_gpu.unsqueeze(0)
    real_rayScat = two_value_interpolation(real_gen_radii, real_rayScat, r)
    real_rayScat = 10.0 ** real_rayScat[:,0].unsqueeze(1)

    scatter_values = compute_scattering(x_f_train_norm, network).unsqueeze(1)
    
    phi = (phi * pi) # between -pi and +pi

    norm = torch.abs(torch.cos(phi)) + (torch.abs(torch.sin(phi))*l_x/l_y)

    #res1_1 = (torch.cos(phi)*grad_u1_x*l_y/l_x) + (torch.sin(phi)*grad_u1_y) + (l_y*real_rayScat*u[:,0].unsqueeze(1))
    res1_1 = (torch.cos(phi)*grad_u1_x) + (torch.sin(phi)*grad_u1_y*l_x/l_y) + (l_x*real_rayScat*u[:,0].unsqueeze(1))
    res1_1 = res1_1/norm

    res1_2 = u[:,0].unsqueeze(1) + (torch.cos(phi)*grad_u1_x/(l_x*real_rayScat+1e-5)) + (torch.sin(phi)*grad_u1_y/(l_y*real_rayScat+1e-5))
    
    res1_1 = torch.minimum(abs(res1_1), abs(res1_2))
    # res1_1 = torch.minimum(abs(l_x*res1), abs(l_y*res1))
    # res1_1 = torch.minimum(res1_1, abs(res1/(real_rayScat+1e-5)))
    
    #res2_1 = (torch.cos(phi)*grad_u2_x*l_y/l_x) + (torch.sin(phi)*grad_u2_y) + (l_y*real_rayScat*(u[:,0]+u[:,1]).unsqueeze(1)) - (l_y*real_rayScat*scatter_values)
    res2_1 = (torch.cos(phi)*grad_u2_x) + (torch.sin(phi)*grad_u2_y*l_x/l_y) + (l_x*real_rayScat*(u[:,0]+u[:,1]).unsqueeze(1)) - (l_x*real_rayScat*scatter_values)
    res2_1 = res2_1/norm

    res2_2 = (u[:,0]+u[:,1]).unsqueeze(1) + (torch.cos(phi)*grad_u2_x/(l_x*real_rayScat+1e-5)) + (torch.sin(phi)*grad_u2_y/(l_y*real_rayScat+1e-5)) - (scatter_values)
    
    res2_1 = torch.minimum(abs(res2_1), abs(res2_2))
    # res2_1 = torch.minimum(res2_1, abs(res2/(real_rayScat+1e-5)))
    # res2_1 = (1-torch.sigmoid(real_rayScat)) * abs(res2_1) + torch.sigmoid(real_rayScat) * abs(res2_2)
    
    print(f'{torch.sqrt(torch.mean(res1_1**2)).round(decimals=4).item():.4f}, {torch.sqrt(torch.mean(res2_1**2)).round(decimals=4).item():.4f}')

    #resDiffpos = u[:,1].unsqueeze(1) - u[:,0].unsqueeze(1)
    #resDiffpos[resDiffpos>0.0] = 0.0 # enforce diffuse light to be positive
    
    #res1_1 = torch.cat([res1_1, torch.sqrt(res2_param)*res2_1, torch.sqrt(resDiffpos_param)*resDiffpos], dim=-1)
    res1_1 = torch.cat([res1_1, torch.sqrt(res2_param)*res2_1], dim=-1)

    if resGrad:
        grad_res = torch.autograd.grad(res, x_f_train_norm, grad_outputs=torch.ones(x_f_train_norm.shape[0]).to(dev), create_graph=True)[0]
        res = torch.cat([res.unsqueeze(1), grad_res], dim=-1)
    
    if False:
        max_ind = torch.argmax(abs(res))
        print(res[max_ind].item(), abs(res2[max_ind]).item())
        print((torch.cos(mu)*grad_u_x/torch.sqrt(r0**2-r99**2))[max_ind].item(), (torch.sin(mu)*grad_u_y*2/(r0-r99))[max_ind].item(), (alphas*u)[max_ind].item(), (0.5*real_rayScat*scatter_values)[max_ind].item())
        print(x_f_train_norm[max_ind,:])
        #print(alphas[max_ind].item(), real_rayScat[max_ind].item(), scatter_values[max_ind].item())
        #print(r0[max_ind].item()/r_jup_mean, r99[max_ind].item()/r_jup_mean, mu[max_ind].item()/pi)
        #print(torch.log10(torch.sqrt(r0**2-r99**2)[max_ind]).item(), torch.log10((r0-r99)[max_ind]).item())
        #print(grad_factor[max_ind].item(), grad_factor_inv[max_ind].item())

    return res1_1


def add_internal_points(n_internal):
    """
    Generates internal points initialized to zeros.
    
    Args:
    - n_internal (int): Number of internal points to generate.
    
    Returns:
    - tuple: Tensors of internal points and corresponding values, both initialized to zero.
    """
    x_internal = torch.tensor(()).new_full(size=(n_internal, space_dimensions), fill_value=0.0)
    y_internal = torch.tensor(()).new_full(size=(n_internal, 1), fill_value=0.0)

    return x_internal, y_internal


def add_boundary(n_boundary, seed=1024):
    """
    Generates boundary points based on specified domain values and seed.
    
    Args:
    - n_boundary (int): Number of boundary points to generate.
    - seed (int): Random seed for generating points.
    
    Returns:
    - tuple: Tensors of boundary points and boundary condition values.
    """
    x_boundary = generator_samples(type_of_points, n_boundary, space_dimensions, seed)
    n_single_dim = int(n_boundary / space_dimensions)
    for i in range(space_dimensions):
        n = int(n_single_dim / 2)
        x_boundary[2 * i * n:n * (2 * i + 1), i] = torch.tensor(()).new_full(size=(n,), fill_value=domain_values[i, 0])
        x_boundary[n * (2 * i + 1):2 * n * (i + 1), i] = torch.tensor(()).new_full(size=(n,), fill_value=domain_values[i, 1])
    ub = torch.tensor(()).new_full(size=(int(n_boundary), 1), fill_value=0.0)
    return x_boundary, ub


def add_collocations(n_collocation, seed=1024, extend=1.0):
    """
    Generates collocation points using a generator function.
    
    Args:
    - n_collocation (int): Number of collocation points.
    - seed (int): Random seed for generating points.
    - extend (float): Extension factor for domain range.
    
    Returns:
    - tuple: Tensors of collocation points and uninitialized values.
    """
    inputs = generator_samples(type_of_points_dom, int(n_collocation), space_dimensions, seed, extend=extend)

    u = torch.tensor(()).new_full(size=(n_collocation, 1), fill_value=0.0)
    return inputs, u

def scaled_angle(phi, lx, ly):
    """
    Scales an angle phi based on domain lengths lx and ly.

    Args:
    - phi (Tensor): Angle in radians.
    - lx, ly (float): Lengths in the x and y directions of the domain.

    Returns:
    - Tensor: Scaled angle.
    """
    
    x_prime = torch.cos(phi) / lx
    y_prime = torch.sin(phi) / ly
    
    phi_scaled = torch.atan2(y_prime, x_prime)
    
    return phi_scaled

def apply_BC(x_boundary, u_boundary, model):
    """
    Applies boundary conditions to the model predictions.
    
    Args:
    - x_boundary (Tensor): Boundary points.
    - u_boundary (Tensor): Boundary condition values.
    - model (torch.nn.Module): Neural network model for prediction.
    
    Returns:
    - tuple: Predicted and actual boundary condition values.
    """
    n_boundary = x_boundary.shape[0]
    n_single_dim = int(n_boundary / space_dimensions)
    n = int(n_single_dim / 2)

    # repeat for boundary at ymin and for u(mu=-1) = u(mu=1)
    x_BC = torch.cat([x_boundary[:3*n,:], x_boundary[2*n:5*n,:], x_boundary[4*n:5*n,:]], dim=0) 
    u_BC = torch.cat([u_boundary[:3*n,:], u_boundary[2*n:5*n,:], u_boundary[4*n:5*n,:]], dim=0)

    x = x_BC[:, 0].unsqueeze(1)
    y = x_BC[:, 1].unsqueeze(1)
    phi = x_BC[:, 2].unsqueeze(1)
    
    Delta_star = x_BC[:,6].unsqueeze(1)
    Delta_star_norm = (Delta_star - (0.5*Delta_star_max/pi)) / (0.5*Delta_star_max/pi)
    
    phi[:n,:] = 0.5*phi[:n,:] # xmin
    phi[:n:2,:] = phi[:n:2,:]*Delta_star[:n:2,:]/pi # xmin, abs(mu)<mu_max
    phi[n:2*n:2,:] = abs(0.5*phi[n:2*n:2,:])+0.5 # xmax
    phi[(n+1):2*n:2,:] = abs(0.5*phi[(n+1):2*n:2,:])-1.0 # xmax
    phi[2*n:3*n,:] = abs(phi[2*n:3*n,:]) # ymin
    phi[2*n:3*n:2,:] = phi[2*n:3*n:2,:]*Delta_star[2*n:3*n:2,:]/pi # ymin, abs(mu)<mu_max
    phi[4*n:5*n,:] = -abs(phi[4*n:5*n,:]) # ymax
    phi[4*n:5*n:2,:] = phi[4*n:5*n:2,:]*Delta_star[4*n:5*n:2,:]/pi # ymax, abs(mu)<mu_max
    phi[6*n:7*n,:] = - phi[5*n:6*n,:] # for bc u(mu=-1) = u(mu=1)
    
    r_pl = x_BC[:, 3].unsqueeze(1)
    a = x_BC[:, 4].unsqueeze(1)
    
    real_gen_radii = r_isotherm(a, r_pl)
    r_pl = ((r_pl / r_jup_mean) - 1.05) / 0.95
    a = (torch.log10(a) - 7.1) / 0.35

    r0 = real_gen_radii[:,0].unsqueeze(1)
    r99 = real_gen_radii[:,-1].unsqueeze(1)
    r = radius(x, y, r0, r99)

    lx = r0 * torch.sqrt( 1 - (r99/r0)**2)
    ly = 0.5*(r0-r99)

    # calculate u(x=0,y2,phi-dphi) for bc at u(x0,ymin,phi):
    dphi = torch.atan(abs(x[2*n:3*n,:])*lx[2*n:3*n,:]/r99[2*n:3*n,:])
    x[3*n:4*n,:] = 0.0
    y[3*n:4*n,:] = (r[2*n:3*n,:] - r99[2*n:3*n,:]) / ly[2*n:3*n,:] - 1.0
    phi[3*n:4*n,:] = phi[2*n:3*n,:] - dphi
    Delta_star_norm[3*n:4*n,:] = 1.0
    
    rayScat_train = x_BC[:,5].unsqueeze(1)

    # u inside abs(phi) < Delta_star is 1, else 0
    u_BC[abs(phi)<=Delta_star] = 1.0
    
    x_BC_norm = torch.cat([x, y, phi, r_pl, a, rayScat_train, Delta_star_norm], dim=-1).float()
    condition = torch.logical_or(x[2*n:3*n,:] > 0, abs(phi[2*n:3*n,:]) > Delta_star[2*n:3*n,:])
    x_BC_norm[3*n:4*n,:] = torch.where(condition, x_BC_norm[2*n:3*n,:], x_BC_norm[3*n:4*n,:])
    #x_BC_norm[3*n:4*n,:] = x_BC_norm[2*n:3*n,:]
    #x_BC_norm.requires_grad = True
    
    u_pred = model(x_BC_norm)#[:,1].unsqueeze(1)
    #print(torch.sum(torch.isnan(u_pred)))
    
    # u(x <= 0,ymin,phi) = u(x2,y2,-phi) or u(x>0,ymin,phi) = 0
    u_BC = u_BC.repeat(1,2)
    u_BC[:,1] = 0.0 # scattered light should be zero at the boundaries


    u_BC[2*n:3*n,0] = torch.where(condition[:,0], 0.0, u_pred[3*n:4*n,0]) #.detach())
    condition = torch.logical_and(~condition[:,0], x_BC_norm[3*n:4*n,2] < -Delta_star_max/pi)
    #print('fulfill condition:', torch.sum(condition))
    u_BC[2*n:3*n,0] = torch.where(condition, 1.0, u_BC[2*n:3*n,0])
    u_BC[2*n:3*n,0] *= torch.sqrt(resDiffBC_param)
    #print(u_BC.shape, u_pred.shape, resDiff_param.shape)
    u_pred[2*n:3*n,1] = torch.sqrt(resDiffBC_param) * torch.sqrt(torch.sin(phi[2*n:3*n,0]*pi)) * u_pred[2*n:3*n,1]
    #u_pred[2*n:3*n,:] = torch.sqrt(resDiffBC_param) * torch.sqrt(torch.sin(phi[2*n:3*n,:]*pi)) * u_pred[2*n:3*n,:]
    #u_BC[2*n:3*n,0] = torch.where(x[2*n:3*n,0] > 0, torch.sqrt(resDiffBC_param) * torch.sqrt(torch.sin(phi[2*n:3*n,0]*pi)) * u_BC[2*n:3*n,1], u_BC[2*n:3*n,0])
    u_pred[2*n:3*n,0] = torch.where(x[2*n:3*n,0] > 0, torch.sqrt(resDiffBC_param) * torch.sqrt(torch.sin(phi[2*n:3*n,0]*pi)) * u_pred[2*n:3*n,0], torch.sqrt(resDiffBC_param) * u_pred[2*n:3*n,0])
    #print(u_BC.shape, u_pred.shape, resDiff_param.shape)
    
    u_BC[:,1] = torch.sqrt(res2_param) * u_BC[:,1]
    u_pred[:,1] = torch.sqrt(res2_param) * u_pred[:,1]
    
    u_BC[5*n:6*n,:] = u_pred[6*n:7*n,:] # for bc u(phi=-1) = u(phi=1)

    # exclude ranges [3*n:4*n] and [6*n:7*n]
    u_BC = torch.cat([u_BC[:3*n,:], u_BC[4*n:6*n,:]], dim=0)
    u_pred = torch.cat([u_pred[:3*n,:], u_pred[4*n:6*n,:]], dim=0)

    return u_pred.reshape(-1), u_BC.reshape(-1)


