from ImportFile import *
import time
#from petitRADTRANS import Radtrans

pi = math.pi

extrema_values = None
space_dimensions = 1
time_dimensions = 0
domain_values = torch.tensor([[-1.0, 1.0]]) # x, only boundary condition at x = -1
parameters_values = torch.tensor([[-1.0, 1.0]]) # y, other parameters added with append_rand_int()

type_of_points = "uniform" # only uniform works
type_of_points_dom = "uniform" # only uniform works
input_dimensions = 104 # (x, y, R, a, 100x Alpha_i)
output_dimension = 1

causality = False # Train using causal weights (Wang et al. 2022), see function compute_weights()
caus_iter = 0 # DONT CHANGE, counts the number of epochs when the causal weights are non-zero everywhere
epsilon = 1e-4 # causality parameter

generate_train_data = True # if training data is completely generated on the fly
# (otherwise the a large sample of alpha profiles and radius parameters will be computed beforehand and a random subsample is used during training)
n_before = 1e7 # number of training points that are precomputed and sampled from during training

resGrad = False # if gradient of residual with respect to the inputs is added to the loss (Yu et al. 2021, Gradient-enhanced PINNs [...])
resGradFactor = 0.001

ub_0 = 1.0
r_jup_mean = 6991100000.0

if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device("cpu")

mean_alphas = torch.linspace(-18.0, -6.0, 100, dtype=torch.float32)
std_alphas = torch.linspace(4.5, 4.5, 100, dtype=torch.float32)
mean_alphas_gpu = torch.linspace(-18.0, -6.0, 100, dtype=torch.float32).to(dev)
std_alphas_gpu = torch.linspace(4.5, 4.5, 100, dtype=torch.float32).to(dev)

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
        gen_alphas = get_alphas(n, random_seed=random_seed)
        gen_radii = get_radii(n, random_seed=random_seed)
    else:
        torch.manual_seed(random_seed)
        ind_alphas = torch.randint(alphas_fixed.shape[0], size=(n,))
        gen_alphas = alphas_fixed[ind_alphas,:]
        ind_radii = torch.randint(radii_fixed.shape[0], size=(n,))
        gen_radii = radii_fixed[ind_radii,:]
        
    rand_samples = torch.cat([rand_samples,gen_radii,gen_alphas], dim=1)
    
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
    dy2 = (((y+1)/2 * (r0-r99))+r99)**2
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


def get_alphas(n, random_seed=42):
    """
    Generates alpha values for n samples with a given random seed.

    Args:
    - n (int): Number of samples to generate alpha values for.
    - random_seed (int, optional): Seed for random number generation. Defaults to 42.

    Returns:
    - Tensor: A tensor of generated alpha values.

    The function generates alpha values based on specified distribution conditions. It divides the
    samples into quarters, generates boolean and numerical values based on truncated exponential
    distributions, and manipulates these values to create a structured array of alphas.
    The function also includes specific transformations and selections to generate the final set of alphas.
    """
    torch.manual_seed(random_seed)
    
    if True:
        n3 = int(n/4)
        n2 = int(np.round(n/4 + 0.001))
        n1 = int(np.ceil(n/4))

        # Generate boolean and numerical values for alphas
        rand_bool1 = torch.randint(0, 2, size=(n1+n3, 99), dtype=torch.bool)
        rand_bool2 = torch.randint(0, 2, size=(n3, 99), dtype=torch.bool)
        rand_nums1 = ((rand_bool1*2-1)) * torch.tensor(truncexpon.rvs(3 * torch.e, loc=0.0, scale=1.0 / torch.e / 6, size=(n1+n3, 99)), dtype=torch.float32)
        rand_nums2 = -torch.tensor(truncexpon.rvs(3 * torch.e, loc=0.0, scale=1.0 / torch.e / 6, size=(n2+n3, 99)), dtype=torch.float32) + 1.01
        rand_nums1 = (rand_nums1 + 1.0) / 99 * 8
        rand_nums2 = (rand_nums2 + 1.0) / 99 * 8
        #rand_nums1[::3,:] = 1.0 / 99 * 8
        #rand_nums2[::3,:] = 2.0 / 99 * 8

        # Further manipulation to generate alphas
        rand_nums3 = torch.zeros((n3,99), dtype=torch.float32)
        rand_nums3[rand_bool2] = rand_nums1[1::2][rand_bool2]
        rand_nums3[~rand_bool2] = rand_nums2[1::2][~rand_bool2]
        rand_nums3[rand_nums3 < (0.8/99*8)] = (torch.rand(rand_nums3[rand_nums3 < (0.8/99*8)].shape)*1.3 - 0.5) / 99 * 8
        rand_nums3[::3,:] = (torch.rand(rand_nums3[::3,:].shape) + 1.0) / 99 * 8

        # Assemble alphas from generated components
        alphas = torch.zeros((n, 100))
        alphas[:, 49] = torch.rand((n,), dtype=torch.float32) * (15.0) - 20.0
        alphas[0::4, :49] = alphas[0::4,49][:, None] - torch.cumsum(rand_nums1[::2,:49], dim=-1).flip(-1)
        alphas[0::4, 50:] = alphas[0::4,49][:, None] + torch.cumsum(rand_nums1[::2,49:], dim=-1)
        alphas[1::4, :49] = alphas[1::4,49][:, None] - torch.cumsum(rand_nums2[::2,:49], dim=-1).flip(-1)
        alphas[1::4, 50:] = alphas[1::4,49][:, None] + torch.cumsum(rand_nums2[::2,49:], dim=-1)
        alphas[3::4, :49] = alphas[3::4,49][:, None] - torch.cumsum(rand_nums3[:,:49], dim=-1).flip(-1)
        alphas[3::4, 50:] = alphas[3::4,49][:, None] + torch.cumsum(rand_nums3[:,49:], dim=-1)

        # Specific transformations for a subset of alphas
        alphas[2::4,:] = torch.log10((10**alphas[0::4,:] + 10**alphas[1::4,:])) - 0.5
        alphas_selected = alphas[2::4,:]
        alphas_selected[::2,:] = (alphas_selected[::2,:] - torch.linspace(-18.0, -6.0, 100)) / torch.linspace(4.5, 4.5, 100)
        alphas_selected[::2,:] = (-alphas_selected[::2,:] * torch.linspace(4.5, 4.5, 100)) + torch.linspace(-18.0, -6.0, 100) - 0.5
        alphas[2::4,:] = alphas_selected
        
        #alphas_selected = alphas[::101,:]
        #for i in range(100):
            #alphas_selected[i::100,:i] = (-1.5 * std_alphas[:i]) + mean_alphas[:i]
        #alphas[::101,:] = alphas_selected
        
        
    else:
        rand_choice = torch.randint(0, n_alphas, size=(n,), dtype=torch.int32)
        rand_d_alphas = d_train_alphas[rand_choice,:]
        rand_d_alphas = rand_d_alphas + (torch.normal(0.0, 0.02/99*8, size=rand_d_alphas.shape)) # augment
        
        alphas = torch.zeros((n, 100))
        alphas[:, 49] = torch.rand((n,), dtype=torch.float32) * (15.0) - 20.0
        alphas[:, :49] = alphas[:,49][:, None] - torch.cumsum(rand_d_alphas[:,:49], dim=-1).flip(-1)
        alphas[:, 50:] = alphas[:,49][:, None] + torch.cumsum(rand_d_alphas[:,49:], dim=-1)

    #mean_alphas = torch.linspace(-18.0, -6.0, 100, dtype=torch.float32)
    #std_alphas = torch.linspace(6 / 2, 10 / 2, 100, dtype=torch.float32)
    #alphas = (alphas - mean_alphas) / std_alphas
    #alphas[alphas<-1.5] = -1.5
    #alphas = (alphas * std_alphas) + mean_alphas

    return alphas

def get_radii(n, random_seed=42):
    """
    Generates radii and scale height values for n samples.

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
    # pre-calculate large sample (n_before) of alpha profiles and radius parameters for faster training
    # torch.manual_seed(42)
    print('generating training data ...')
    alphas_fixed = get_alphas(int(n_before))
    print('alpha shape: ', alphas_fixed.shape)
    # sort out high alphas at the outermost layer (ensure smooth transition between empty space and atmosphere)
    alphas_fixed = alphas_fixed[alphas_fixed[:,0]<(-10.0), :]
    # sort out low alphas at innermost layer where the solution is trivial (u = 1 everywhere)
    alphas_fixed = alphas_fixed[alphas_fixed[:,-1]>(-13.0), :]
    print('alpha shape: ', alphas_fixed.shape)
    radii_fixed = get_radii(int(n_before))
    print('radii shape: ', radii_fixed.shape)
    real_radii_fixed = r_isotherm(radii_fixed[:,1].unsqueeze(1), radii_fixed[:,0].unsqueeze(1), gpu=False)[:,0]
    # sort out r-profiles that don't work with the anayltical formula in r_isotherm()
    radii_fixed = radii_fixed[torch.logical_and(real_radii_fixed>0.0, real_radii_fixed<3.0*r_jup_mean),:]
    print('radii shape: ', radii_fixed.shape)

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

    extrema = torch.cat([domain_values, parameters_values], 0)
    extrema_0 = extrema[:, 0]
    extrema_f = extrema[:, 1]

    extrema_f[0] = extend * (extrema_f[0] - extrema_0[0]) + extrema_0[0]

    if type_point_param == "uniform":
        if random_seed is not None:
            torch.random.manual_seed(random_seed)
        params = torch.rand([samples, dim]).type(torch.FloatTensor) * (extrema_f - extrema_0) + extrema_0
        params = append_rand_int(params, dim, random_seed)
        return params
    elif type_point_param == "sobol":
        # if n_time_step is None:
        skip = random_seed
        data = np.full((samples, dim), np.nan)
        for j in range(samples):
            seed = j + skip
            data[j, :], next_seed = sobol_seq.i4_sobol(dim, seed)
            # data[j, :-help_param], next_seed = sobol_seq.i4_sobol(dim-help_param, seed)
        params = torch.from_numpy(data).type(torch.FloatTensor) * (extrema_f - extrema_0) + extrema_0
        params = append_rand_int(params, dim, random_seed)
        return params
    elif type_point_param == "grid":
        # if n_time_step is None:
        if dim == 2:
            n_mu = 128
            n_x = int(samples / n_mu)
            x = np.linspace(0, 1, n_x + 2)
            mu = np.linspace(0, 1, n_mu)
            x = x[1:-1]
            inputs = torch.from_numpy(np.transpose([np.repeat(x, len(mu)), np.tile(mu, len(x))])).type(torch.FloatTensor)
            inputs = inputs * (extrema_f - extrema_0) + extrema_0
        elif dim == 1:
            x = torch.linspace(0, 1, samples).reshape(-1, 1)
            mu = torch.linspace(0, 1, samples).reshape(-1, 1)
            inputs = torch.cat([x, mu], 1)
            inputs = inputs * (extrema_f - extrema_0) + extrema_0
        else:
            raise ValueError()
        inputs = calc_help_param(inputs)
        return inputs.to(dev)


def compute_res(network, x_f_train, space_dimensions, solid_object, computing_error, max_alpha=100.0):
    """
    Computes the residual of the network's predictions against physical laws (RTE).
    
    Args:
    - network (torch.nn.Module): The neural network model.
    - x_f_train (Tensor): Training data inputs.
    - space_dimensions (int): Number of spatial dimensions in the data.
    - solid_object: -
    - computing_error: -
    - max_alpha (float): Maximum value of alpha for scaling. [DEPRECATED]
    
    Returns:
    - Tensor: The computed residuals.
    """
    x = x_f_train[:, 0].unsqueeze(1)
    y = x_f_train[:, 1].unsqueeze(1)
    r_pl = x_f_train[:, 2].unsqueeze(1)
    a = x_f_train[:, 3].unsqueeze(1)

    
    real_gen_radii = r_isotherm(a, r_pl)
    r_pl = ((r_pl / r_jup_mean) - 1.05) / 0.95
    a = (torch.log10(a) - 7.1) / 0.35

    real_gen_alpha = x_f_train[:,4:]
    # alpha_train = torch.log10((10**real_gen_alpha) * torch.sqrt(real_gen_radii[:,0]**2 - real_gen_radii[:,-1]**2).unsqueeze(1))
    alpha_train = (real_gen_alpha - mean_alphas_gpu) / std_alphas_gpu

    x_f_train_norm = torch.cat([x, y, r_pl, a, alpha_train], dim=-1).float()
    x_f_train_norm.requires_grad = True

    u = network(x_f_train_norm).reshape(-1)
    grad_u = torch.autograd.grad(u, x_f_train_norm, grad_outputs=torch.ones(x_f_train_norm.shape[0]).to(dev), create_graph=True)[0]

    grad_u_x = grad_u[:, 0]

    r = radius(x, y, real_gen_radii[:,0].unsqueeze(1), real_gen_radii[:,-1].unsqueeze(1))

    alphas = two_value_interpolation(real_gen_radii, real_gen_alpha, r)
    alphas = 10.0 ** alphas
    alphas *= torch.sqrt(real_gen_radii[:,0] ** 2 - real_gen_radii[:,-1] ** 2).unsqueeze(1)
    alphas = alphas[:, 0]

    residual1 = grad_u_x + (alphas * u)
    residual2 = (grad_u_x / (alphas + 1e-5)) + u

    res = torch.minimum(abs(residual1), abs(residual2))

    if resGrad:
        grad_res = torch.autograd.grad(res, x_f_train_norm, grad_outputs=torch.ones(x_f_train_norm.shape[0]).to(dev), create_graph=True)[0]
        res = torch.cat([res.unsqueeze(1), grad_res], dim=-1)

    return res


def compute_weights(x_f_train, res,eps=epsilon):
    # attributes weights to the training points according to causality (Wang et al. 2022)
    
    x = x_f_train[:, 0]
    order = torch.argsort(x, dim=0).to(dev)
    weights = torch.ones(res.shape[0], dtype=torch.double).to(dev)
    weights[order[1:]] = torch.exp(-eps * torch.cumsum(res[order], 0).detach()[:-1])
    # print(torch.exp(-eps * torch.cumsum(res[order], 0)[:-1])[-5:])
    # print(torch.min(weights), torch.max(weights))
    current_x = x[torch.argmin(abs(weights-0.5))].item()
    print(current_x, eps)

    global caus_iter
    if current_x > 0.99 and eps < 0.003:
    	caus_iter += 1
    else:
        caus_iter = 0

    if caus_iter >= 21:
        eps = eps * 10**0.5

    return weights, eps


def add_internal_points(n_internal):
    """
    Generates internal points initialized to zeros.
    
    Args:
    - n_internal (int): Number of internal points to generate.
    
    Returns:
    - tuple: Tensors of internal points and corresponding values, both initialized to zero.
    """
    x_internal = torch.tensor(()).new_full(size=(n_internal, 2), fill_value=0.0)
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
    x_boundary = generator_samples(type_of_points, n_boundary, 2, seed)
    n_single_dim = int(n_boundary / space_dimensions)
    for i in range(space_dimensions):
        n = int(n_single_dim / 2)
        x_boundary[2 * i * n:n * (2 * i + 1), i] = torch.tensor(()).new_full(size=(n,), fill_value=domain_values[i, 0])
        x_boundary[n * (2 * i + 1):2 * n * (i + 1), i] = torch.tensor(()).new_full(size=(n,), fill_value=domain_values[i, 1])
    ub = torch.tensor(()).new_full(size=(int(n_boundary), 1), fill_value=ub_0)
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
    inputs = generator_samples(type_of_points_dom, int(n_collocation), 2, seed, extend=extend)

    u = torch.tensor(()).new_full(size=(n_collocation, 1), fill_value=np.nan)
    return inputs, u


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
    where_x_equal_0 = x_boundary[:, 0] == domain_values[0, 0]
    x_BC = x_boundary[where_x_equal_0, :]
    u_BC = u_boundary[where_x_equal_0, :]

    x = x_BC[:, 0].unsqueeze(1)
    y = x_BC[:, 1].unsqueeze(1)
    r_pl = x_BC[:, 2].unsqueeze(1)
    a = x_BC[:, 3].unsqueeze(1)
    
    real_gen_radii = r_isotherm(a, r_pl)
    r_pl = ((r_pl / r_jup_mean) - 1.05) / 0.95
    a = (torch.log10(a) - 7.1) / 0.35

    real_gen_alpha = x_BC[:,4:]
    # alpha_train = torch.log10((10**real_gen_alpha) * torch.sqrt(real_gen_radii[:,0]**2 - real_gen_radii[:,-1]**2).unsqueeze(1))
    alpha_train = (real_gen_alpha - mean_alphas_gpu) / std_alphas_gpu

    x_BC_norm = torch.cat([x, y, r_pl, a, alpha_train], dim=-1).float()

    u_pred = model(x_BC_norm)

    return u_pred.reshape(-1), u_BC.reshape(-1)
