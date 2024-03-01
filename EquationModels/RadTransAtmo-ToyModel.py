from ImportFile import *
import time

pi = math.pi

extrema_values = None
space_dimensions = 1
time_dimensions = 0
domain_values = torch.tensor([[0.0, 1.0]])
parameters_values = torch.tensor([[0.0, 1.0],  # y
                                  [0.2, 0.8],  # temperature
                                  [-10.0, -3.0]]) # log(sum(10^frac_i * sigma_i))

type_of_points = "uniform"
type_of_points_dom = "uniform"
input_dimensions = 4
help_param = 0
output_dimension = 1

ub_0 = 1.0
#strip = 0.05

n_quad = 10

R_pl = 100
T_mean = torch.mean(parameters_values[1,:])
l_x = torch.sqrt(2*R_pl / (0.1*T_mean))

# Parameters for the two species
n_species = 2
nu0 = torch.tensor([[0.3], [0.6]])
sigma0 = torch.tensor([[8e-4], [4e-4]])

if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device("cpu")

def z(x,y):
    dx = l_x * (x-0.5)
    dy = y + R_pl
    return torch.sqrt(dx**2 + dy**2) - R_pl

def cross_section(nu, T):
    if nu.dim() == 0:
        sigmas = torch.zeros(1, n_species)
    else:
        sigmas = torch.zeros(nu.size(dim=0), n_species)
    for i in range(n_species):
        for j in range(nu0[i].size(dim=0)):
            sigmas[:,i] = sigmas[:,i] + (sigma0[i,j] / torch.sqrt(pi*T) * torch.exp(- (nu-nu0[i,j])**2 / ((0.2*nu0[i,j])**2 * T)))
    return sigmas

def density(z, T):
    H_P = 0.1 * T
    return 1e6 * torch.exp(-z/H_P)
    
def calc_help_param(samples):
    if help_param == 0:
        return samples
    else:
        #mask = samples[:,0] > 0.5
        #samples[mask, 4] = samples[mask,1] 
        #samples[~mask,4] = z(samples[~mask,0], samples[~mask,1]) # give z as an additional input
        samples[:,4] = z(samples[:,0], samples[:,1])
        return samples
    
help_values = torch.tensor([[0.0, z(1.0,1.0)]])

def generator_samples(type_point_param, samples, dim, random_seed):
    if help_param == 0:
        extrema = torch.cat([domain_values, parameters_values], 0)
    else:
        extrema = torch.cat([domain_values, parameters_values, torch.zeros((help_param, 2))], 0)
        # dim = dim - help_param
    extrema_0 = extrema[:, 0]
    extrema_f = extrema[:, 1]
    if type_point_param == "uniform":
        if random_seed is not None:
            torch.random.manual_seed(random_seed)
        params = torch.rand([samples, dim]).type(torch.FloatTensor) * (extrema_f - extrema_0) + extrema_0
        params = calc_help_param(params)
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
        params = calc_help_param(params)
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
    x_f_train.requires_grad = True
    x = x_f_train[:, 0]
    y = x_f_train[:, 1]
    T = x_f_train[:, 2]
    frac_sigma = 10**x_f_train[:, 3]
    
    x_f_train_norm = normalize(x_f_train)
    u = network(x_f_train_norm).reshape(-1, )
    grad_u = torch.autograd.grad(u, x_f_train_norm, grad_outputs=torch.ones(x_f_train_norm.shape[0], ).to(dev), create_graph=True)[0]

    grad_u_x = grad_u[:, 0]
    
    if help_param != 0:
    
        dr = x_f_train[:, 4]
        
        grad_u_z = grad_u[:, 4]
        grad_z_x = l_x**2 * (x - 0.5) / (dr + R_pl)
        
        grad_u_x = grad_u_z * grad_z_x + grad_u_x
        
    else:
    
        dr = z(x, y)
        
    n = density(dr, T)
    # n = n.clone().detach().reshape(-1,1)
    # n = n.expand(-1, 2)
    
    alpha = l_x * n * frac_sigma
    # alpha_clipped = torch.minimum(alpha, torch.tensor([max_alpha]))
    
    residual1 = grad_u_x + (alpha * u)
    residual2 = (grad_u_x/(alpha+1e-5)) + u
    
    weights = 1.0 # torch.log(2.0 + alpha)
    
    res = weights * torch.minimum(abs(residual1), abs(residual2))

    return res


def add_internal_points(n_internal):
    x_internal = torch.tensor(()).new_full(size=(n_internal, input_dimensions), fill_value=0.0)
    y_internal = torch.tensor(()).new_full(size=(n_internal, 1), fill_value=0.0)

    return x_internal, y_internal


def add_boundary(n_boundary, seed=1024):
    x_boundary = generator_samples(type_of_points, n_boundary, input_dimensions, seed)
    n_single_dim = int(n_boundary / space_dimensions)
    for i in range(space_dimensions):
        n = int(n_single_dim / 2)
        x_boundary[2 * i * n:n * (2 * i + 1), i] = torch.tensor(()).new_full(size=(n,), fill_value=domain_values[i, 0])
        x_boundary[n * (2 * i + 1):2 * n * (i + 1), i] = torch.tensor(()).new_full(size=(n,), fill_value=domain_values[i, 1])
    ub = torch.tensor(()).new_full(size=(int(n_boundary), 1), fill_value=ub_0)
    return x_boundary, ub


def add_collocations(n_collocation, seed=1024):
    inputs = generator_samples(type_of_points_dom, int(n_collocation), input_dimensions, seed)

    u = torch.tensor(()).new_full(size=(n_collocation, 1), fill_value=np.nan)
    return inputs, u


def apply_BC(x_boundary, u_boundary, model):

    where_x_equal_0 = x_boundary[:, 0] == domain_values[0, 0]
    # where_y_equal_1 = x_boundary[:, 1] == domain_values[1, 1]
    # mask = torch.logical_or(where_x_equal_0, where_y_equal_1)

    x_BC = x_boundary[where_x_equal_0, :]
    u_BC = u_boundary[where_x_equal_0, :]

    x_BC_norm = normalize(x_BC)
    u_pred = model(x_BC_norm)

    return u_pred.reshape(-1, ), u_BC.reshape(-1, )


def normalize(vector):
    if help_param == 0:
        max_val, _ = torch.max(torch.cat([domain_values, parameters_values], 0), 1)
        min_val, _ = torch.min(torch.cat([domain_values, parameters_values], 0), 1)
    else:
        max_val, _ = torch.max(torch.cat([domain_values, parameters_values, help_values], 0), 1)
        min_val, _ = torch.min(torch.cat([domain_values, parameters_values, help_values], 0), 1)
    vector = vector * (max_val - min_val) + min_val
    return vector
    
def denormalize(vector):
    vector = np.array(vector)
    max_val = np.max(np.array(torch.cat([domain_values, parameters_values], 0)), axis=1)
    min_val = np.min(np.array(torch.cat([domain_values, parameters_values], 0)), axis=1)
    vector = (vector - min_val) / (max_val - min_val)
    return torch.from_numpy(vector).type(torch.FloatTensor)
