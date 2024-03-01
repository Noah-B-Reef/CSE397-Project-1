from ImportFile import *
import time

pi = math.pi

extrema_values = None
space_dimensions = 3
time_dimensions = 0
domain_values = torch.tensor([[-1.0,1.0], [-1.0,1.0], [-1.0,1.0]]) # x,y,mu
parameters_values = None # torch.tensor([[-1.0, 1.0]]) # y, mu

type_of_points = "uniform"
type_of_points_dom = "uniform"
input_dimensions = 3 # or 204 (x,y,r0,r99,r_i,alpha_i)
output_dimension = 1
causality = False
caus_iter = 0
epsilon = 1e-4
generate_train_data = False
resGrad = False
resGradFactor = 0.001

ub_0 = 1.0
#strip = 0.05
r_jup_mean = 6991100000.0

n_quad = 16

if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device("cpu")

res2_param = torch.tensor([2.0]).to(dev)

mean_rayScat = torch.linspace(-16.7, -8.7, 100, dtype=torch.float32)
mean_rayScat_gpu = torch.linspace(-16.7, -8.7, 100, dtype=torch.float32).to(dev)
std_rayScat = 3.0
mean_alphas = torch.linspace(-18.0, -6.0, 100, dtype=torch.float32)
std_alphas = torch.linspace(4.5, 4.5, 100, dtype=torch.float32)
mean_alphas_gpu = torch.linspace(-18.0, -6.0, 100, dtype=torch.float32).to(dev)
std_alphas_gpu = torch.linspace(4.5, 4.5, 100, dtype=torch.float32).to(dev)

def two_value_interpolation(x, y, val):
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
    n = rand_samples.shape[0]
    
    if generate_train_data:
        gen_alphas = get_alphas(n, random_seed=random_seed)
        gen_radii = get_radii(n, random_seed=random_seed)
        gen_rayScat = get_rayScat(n, random_seed=random_seed)
    else:
        torch.manual_seed(random_seed)
        ind_alphas = torch.randint(alphas_fixed.shape[0], size=(n,))
        gen_alphas = alphas_fixed[ind_alphas,:]
        ind_radii = torch.randint(radii_fixed.shape[0], size=(n,))
        gen_radii = radii_fixed[ind_radii,:]
        gen_rayScat = get_rayScat(n, random_seed=random_seed)
        
    rand_samples = torch.cat([rand_samples,gen_radii,gen_rayScat,gen_alphas], dim=1)
    
    return rand_samples

def radius(x,y,r0,r99):
    dx2 = (x**2) * (r0**2 - r99**2)
    dy2 = (((y+1)/2 * (r0-r99))+r99)**2
    return torch.sqrt(dx2 + dy2)

def r_isotherm(a, r_P0, gpu=True):
    P0 = 0.01
    if gpu:
        pressures = torch.logspace(-6, 2, 100).unsqueeze(0).to(dev)
    else:
        pressures = torch.logspace(-6, 2, 100).unsqueeze(0)
    return r_P0 / (1 + ((a/r_P0)*torch.log(pressures/P0)))

def scattering_phase_fct(mu, mu_prime):
    phi = 0.75*(1 + (mu**2)*(mu_prime**2) + 0.5*(1-mu**2)*(1-mu_prime**2))
    return phi

def compute_scattering(x, model):
    n_quad = torch.randint(16,20,size=(1,)).item()
    mu_prime = torch.linspace(-1,1,n_quad).to(dev)
    
    mu = x[:,2]

    #x_l = list(x[:,:2].detach().cpu().numpy())
    #mu_prime_l = list(mu_prime.detach().cpu().numpy())
    #x_l_rest = list(x[:,3:].detach().cpu().numpy())
    
    x_l = x[:,:2]
    x_l_rest = x[:,3:]

    inputs = [torch.repeat_interleave(x_l, mu_prime.shape[0]).reshape(2,-1), mu_prime.repeat(1,x_l.shape[0]), torch.repeat_interleave(x_l_rest, mu_prime.shape[0]).reshape(103,-1)]
    inputs = torch.transpose(torch.cat(inputs, dim=0), 0, 1)

    u = model(inputs)[:,1].unsqueeze(1)
    u = u.reshape(x.shape[0], mu_prime.shape[0])
    
    mu = torch.cos(mu*pi)
    mu_prime = torch.cos(mu_prime*pi)

    kernel = scattering_phase_fct(mu.unsqueeze(1), mu_prime.unsqueeze(0))

    scatter_values = torch.trapezoid(kernel*u, x=mu_prime, dim=-1)

    return scatter_values.to(dev)

def get_alphas(n, random_seed=42):
    torch.manual_seed(random_seed)

    n3 = int(n/4)
    n2 = int(np.round(n/4 + 0.001))
    n1 = int(np.ceil(n/4))

    # rand_choice = torch.randint(0, 4, size=(n,), dtype=torch.int32)
    rand_bool1 = torch.randint(0, 2, size=(n1+n3, 99), dtype=torch.bool)
    rand_bool2 = torch.randint(0, 2, size=(n3, 99), dtype=torch.bool)
    rand_nums1 = ((rand_bool1*2-1)) * torch.tensor(truncexpon.rvs(3 * torch.e, loc=0.0, scale=1.0 / torch.e / 6, size=(n1+n3, 99)), dtype=torch.float32)
    rand_nums2 = -torch.tensor(truncexpon.rvs(3 * torch.e, loc=0.0, scale=1.0 / torch.e / 6, size=(n2+n3, 99)), dtype=torch.float32) + 1.01
    rand_nums1 = (rand_nums1 + 1.0) / 99 * 8
    rand_nums2 = (rand_nums2 + 1.0) / 99 * 8
    #rand_nums1[::3,:] = 1.0 / 99 * 8
    #rand_nums2[::3,:] = 2.0 / 99 * 8

    rand_nums3 = torch.zeros((n3,99), dtype=torch.float32)
    rand_nums3[rand_bool2] = rand_nums1[1::2][rand_bool2]
    rand_nums3[~rand_bool2] = rand_nums2[1::2][~rand_bool2]
    rand_nums3[rand_nums3 < (0.8/99*8)] = (torch.rand(rand_nums3[rand_nums3 < (0.8/99*8)].shape)*1.3 - 0.5) / 99 * 8
    rand_nums3[::3,:] = (torch.rand(rand_nums3[::3,:].shape) + 1.0) / 99 * 8

    alphas = torch.zeros((n, 100))
    alphas[:, 49] = torch.rand((n,), dtype=torch.float32) * (15.0) - 20.0
    alphas[0::4, :49] = alphas[0::4,49][:, None] - torch.cumsum(rand_nums1[::2,:49], dim=-1).flip(-1)
    alphas[0::4, 50:] = alphas[0::4,49][:, None] + torch.cumsum(rand_nums1[::2,49:], dim=-1)
    alphas[1::4, :49] = alphas[1::4,49][:, None] - torch.cumsum(rand_nums2[::2,:49], dim=-1).flip(-1)
    alphas[1::4, 50:] = alphas[1::4,49][:, None] + torch.cumsum(rand_nums2[::2,49:], dim=-1)
    alphas[3::4, :49] = alphas[3::4,49][:, None] - torch.cumsum(rand_nums3[:,:49], dim=-1).flip(-1)
    alphas[3::4, 50:] = alphas[3::4,49][:, None] + torch.cumsum(rand_nums3[:,49:], dim=-1)

    alphas[2::4,:] = torch.log10((10**alphas[0::4,:] + 10**alphas[1::4,:])) - 0.5
    alphas_selected = alphas[2::4,:]
    alphas_selected[::2,:] = (alphas_selected[::2,:] - torch.linspace(-18.0, -6.0, 100)) / torch.linspace(4.5, 4.5, 100)
    alphas_selected[::2,:] = (-alphas_selected[::2,:] * torch.linspace(4.5, 4.5, 100)) + torch.linspace(-18.0, -6.0, 100) - 0.5
    alphas[2::4,:] = alphas_selected

    # mean_alphas = torch.linspace(-18.0, -6.0, 100, dtype=torch.float32)
    # std_alphas = torch.linspace(6 / 2, 10 / 2, 100, dtype=torch.float32)
    # alphas_norm = (alphas - mean_alphas) / std_alphas

    return alphas

def get_radii(n, random_seed=42):
    torch.manual_seed(random_seed)
    
    radii = torch.zeros((n, 2))
    
    R_pl = (torch.rand((n,)) * 1.8 + 0.2) * r_jup_mean
    a = 10**torch.normal(7.1,0.35, size=(n,))
    
    radii[:,0] = R_pl
    radii[:,1] = a
    
    return radii

if not generate_train_data:
    torch.manual_seed(42)
    alphas_fixed = get_alphas(int(1e7))
    print('alpha shape: ', alphas_fixed.shape)
    alphas_fixed = alphas_fixed[alphas_fixed[:,0]<(-12.5), :]
    alphas_fixed = alphas_fixed[alphas_fixed[:,-1]>(-12.5), :]
    print('alpha shape: ', alphas_fixed.shape)
    radii_fixed = get_radii(int(1e7))
    print('radii shape: ', radii_fixed.shape)
    real_radii_fixed = r_isotherm(radii_fixed[:,1].unsqueeze(1), radii_fixed[:,0].unsqueeze(1), gpu=False)[:,0]
    radii_fixed = radii_fixed[torch.logical_and(real_radii_fixed>0.0, real_radii_fixed<3.0*r_jup_mean),:]
    print('radii shape: ', radii_fixed.shape)


def get_rayScat(n, random_seed=42):

    alpha_rayScat = torch.rand((n,1)) * 3.0 - 1.5
    # alpha_rayScat = alpha_rayScat * std_rayScat + mean_rayScat[-1]
    
    return alpha_rayScat

def generator_samples(type_point_param, samples, dim, random_seed, extend=1.0):

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
    x = x_f_train[:, 0].unsqueeze(1)
    y = x_f_train[:, 1].unsqueeze(1)
    mu = x_f_train[:, 2].unsqueeze(1)
    r_pl = x_f_train[:, 3].unsqueeze(1)
    a = x_f_train[:, 4].unsqueeze(1)

    
    real_gen_radii = r_isotherm(a, r_pl)
    r_pl = ((r_pl / r_jup_mean) - 1.05) / 0.95
    a = (torch.log10(a) - 7.1) / 0.35
    
    rayScat_train = x_f_train[:,5].unsqueeze(1)

    real_gen_alpha = x_f_train[:,6:]
    alpha_train = (real_gen_alpha - mean_alphas_gpu) / std_alphas_gpu

    x_f_train_norm = torch.cat([x, y, mu, r_pl, a, rayScat_train, alpha_train], dim=-1).float()
    x_f_train_norm.requires_grad = True

    u = network(x_f_train_norm).reshape(-1,2)
    
    grad_u1 = torch.autograd.grad(u[:,0], x_f_train_norm, grad_outputs=torch.ones(x_f_train_norm.shape[0]).to(dev), create_graph=True)[0]
    
    grad_u1_x = grad_u1[:, 0].unsqueeze(1)
    grad_u1_y = grad_u1[:, 1].unsqueeze(1)
    
    grad_u2 = torch.autograd.grad(u[:,1], x_f_train_norm, grad_outputs=torch.ones(x_f_train_norm.shape[0]).to(dev), create_graph=True)[0]
    
    grad_u2_x = grad_u2[:, 0].unsqueeze(1)
    grad_u2_y = grad_u2[:, 1].unsqueeze(1)

    r0 = real_gen_radii[:,0].unsqueeze(1)
    r99 = real_gen_radii[:,-1].unsqueeze(1)
    r = radius(x, y, r0, r99)
    
    real_rayScat = (rayScat_train * std_rayScat) + mean_rayScat_gpu.unsqueeze(0)
    real_rayScat = two_value_interpolation(real_gen_radii, real_rayScat, r)
    real_rayScat = 10.0 ** real_rayScat[:,0].unsqueeze(1)

    alphas = two_value_interpolation(real_gen_radii, real_gen_alpha, r)
    alphas = (10.0 ** alphas[:,0].unsqueeze(1)) + real_rayScat
    #alphas = real_rayScat # ONLY Rayleigh scattering!!!
    #print(alphas[r[:,0]>real_gen_radii[:,1]].max())

    scatter_values = compute_scattering(x_f_train_norm, network).unsqueeze(1)
    mu = (mu * pi) # between -pi and +pi

    #grad_factor = (2*torch.sqrt(r0**2-r99**2)/(r0-r99)) * (torch.sin(mu)/torch.cos(mu))
    #grad_factor_inv = (0.5*(r0-r99)/torch.sqrt(r0**2-r99**2)) * (torch.cos(mu)/torch.sin(mu))
    
    alpha_weight = (abs(torch.cos(mu))*torch.sqrt(r0**2-r99**2) + abs(torch.sin(mu))*0.5*(r0-r99))
    alphas = alphas * alpha_weight
    real_rayScat = real_rayScat * alpha_weight
    
    res1_1 = (torch.cos(mu)*grad_u1_x) + (torch.sin(mu)*grad_u1_y) + (alphas*u[:,0].unsqueeze(1))
    
    res1_2 = u[:,0].unsqueeze(1) + (torch.cos(mu)*grad_u1_x/(alphas+1e-5)) + (torch.sin(mu)*grad_u1_y/(alphas+1e-5))
    
    res1_1 = torch.minimum(abs(res1_1), abs(res1_2))
    
    res2_1 = (torch.cos(mu)*grad_u2_x) + (torch.sin(mu)*grad_u2_y) + (alphas*u[:,1].unsqueeze(1)) - (0.5*real_rayScat*scatter_values)
    
    res2_2 = u[:,1].unsqueeze(1) + (torch.cos(mu)*grad_u2_x/(alphas+1e-5)) + (torch.sin(mu)*grad_u2_y/(alphas+1e-5)) - (0.5*(real_rayScat/(alphas+1e-5))*scatter_values)
    
    res2_1 = torch.minimum(abs(res2_1), abs(res2_2))
    
    print(f'{torch.sqrt(torch.mean(res1_1**2)).round(decimals=4).item():.4f}, {torch.sqrt(torch.mean(res2_1**2)).round(decimals=4).item():.4f}')
    
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


def compute_weights(x_f_train, res,eps=epsilon):
    x = x_f_train[:, 0]
    order = torch.argsort(x, dim=0).to(dev)
    weights = torch.ones(res.shape, dtype=torch.float).to(dev)
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
    ub = torch.tensor(()).new_full(size=(int(n_boundary), 1), fill_value=0.0)
    return x_boundary, ub


def add_collocations(n_collocation, seed=1024, extend=1.0):
    inputs = generator_samples(type_of_points_dom, int(n_collocation), input_dimensions, seed, extend=extend)

    u = torch.tensor(()).new_full(size=(n_collocation, 1), fill_value=0.0)
    return inputs, u


def apply_BC(x_boundary, u_boundary, model):
    # u(x=-1,y,-pi/4<mu<pi/4)=1
    n_boundary = x_boundary.shape[0]
    n_single_dim = int(n_boundary / space_dimensions)
    n = int(n_single_dim / 2)
    
    x_BC = torch.cat([x_boundary[:2*n,:], x_boundary[3*n:5*n,:], x_boundary[4*n:5*n,:]], dim=0)
    u_BC = torch.cat([u_boundary[:2*n,:], u_boundary[3*n:5*n,:], u_boundary[4*n:5*n,:]], dim=0)
    #x_BC = x_boundary[:n]
    #u_BC = u_boundary[:n]

    x = x_BC[:, 0].unsqueeze(1)
    y = x_BC[:, 1].unsqueeze(1)
    mu = x_BC[:, 2].unsqueeze(1)
    
    mu[:n,:] = 0.5*mu[:n,:] # xmin
    mu[n:2*n:2,:] = abs(0.5*mu[n:2*n:2,:])+0.5 # xmax
    mu[(n+1):2*n:2,:] = abs(0.5*mu[(n+1):2*n:2,:])-1.0 # xmax
    mu[2*n:3*n,:] = -abs(mu[2*n:3*n,:]) # ymax
    mu[4*n:5*n,:] = - mu[3*n:4*n,:] # for bc u(mu=-1) = u(mu=1)
    
    u_BC[abs(mu)<0.25] = 1.0 # u inside abs(mu) < pi/4 is one (typical value for R=1.0R_jup and log(a)=7.1)
    
    r_pl = x_BC[:, 3].unsqueeze(1)
    a = x_BC[:, 4].unsqueeze(1)
    
    real_gen_radii = r_isotherm(a, r_pl)
    r_pl = ((r_pl / r_jup_mean) - 1.05) / 0.95
    a = (torch.log10(a) - 7.1) / 0.35
    
    rayScat_train = x_BC[:,5].unsqueeze(1)

    real_gen_alpha = x_BC[:,6:]
    # alpha_train = torch.log10((10**real_gen_alpha) * torch.sqrt(real_gen_radii[:,0]**2 - real_gen_radii[:,-1]**2).unsqueeze(1))
    alpha_train = (real_gen_alpha - mean_alphas_gpu) / std_alphas_gpu

    x_BC_norm = torch.cat([x, y, mu, r_pl, a, rayScat_train, alpha_train], dim=-1).float()

    u_pred = model(x_BC_norm)#[:,1].unsqueeze(1)
    
    u_BC = u_BC.repeat(1,2)
    u_BC[:,1] = torch.sqrt(res2_param) * u_BC[:,1]
    u_pred[:,1] = torch.sqrt(res2_param) * u_pred[:,1]
    
    u_BC[3*n:4*n,:] = u_pred[4*n:5*n,:] # for bc u(mu=-1) = u(mu=1)
    u_BC = u_BC[:4*n,:]
    u_pred = u_pred[:4*n,:]
    #u_BC = torch.cat([u_BC[:n,:], u_BC[2*n:3*n,:]], dim=0)
    #u_pred = torch.cat([u_pred[:n,:], u_pred[2*n:3*n,:]], dim=0)

    return u_pred.reshape(-1), u_BC.reshape(-1)


def compute_generalization_error(model, extrema, images_path=None):
    return 0, 0


def plotting(model, images_path, extrema, solid):
    model.cpu()
    model = model.eval()
    n = 500

    x = np.linspace(domain_values[0, 0], domain_values[0, 1], n)
    y = np.linspace(domain_values[1, 0], domain_values[1, 1], n)

    inputs = torch.from_numpy(np.transpose([np.repeat(x, len(y)), np.tile(y, len(x))])).type(torch.FloatTensor)
    print(inputs.size())
    param_vals = torch.tensor([0.3, 0.5, -1, -1]) # freq, T, dens1, dens2
    for i in range(4):
        param = torch.tensor(()).new_full(size=(n*n,1), fill_value=param_vals[i])
        inputs = torch.cat((inputs, param), dim=1)
    inputs = torch.cat((inputs, torch.zeros((n*n, help_param))), dim=1)
    inputs = calc_help_param(inputs)
    inputs_norm = normalize(inputs)
    sol = model(inputs_norm)
    sol = sol.reshape(x.shape[0], y.shape[0])

    x_l = inputs[:, 0]
    y_l = inputs[:, 1]

    #exact = torch.sin(pi * mu_l) ** 2 * torch.cos(pi / 2 * x_l)
    #exact = exact.reshape(x.shape[0], mu.shape[0])

    #print(torch.mean(abs(sol - exact) / torch.mean(abs(exact))))

    levels = [0.00, 0.006, 0.013, 0.021, 0.029, 0.04, 0.047, 0.06, 0.071, 0.099, 0.143, 0.214, 0.286, 0.357, 0.429, 0.5, 0.571, 0.643, 0.714, 0.786, 0.857, 0.929, 1]
    norml = matplotlib.colors.BoundaryNorm(levels, 256)
    plt.figure(1)
    plt.contourf(x.reshape(-1, ), y.reshape(-1, ), sol.detach().numpy().T, cmap='jet', levels=levels, norm=norml)
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'$u(x,y)$')
    plt.savefig(images_path + "/net_sol.png", dpi=400)

