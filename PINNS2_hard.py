from ImportFile import *

pi = math.pi
torch.manual_seed(42)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def initialize_inputs(len_sys_argv):
    if len_sys_argv <= 2:

        if len_sys_argv == 1:
            sampling_seed_ = 42
        else:
            sampling_seed_ = int(sys.argv[1])

        n_coll_ = 32768 * 4 * 4
        n_u_ = 16384 * 4 * 4
        n_int_ = 0

        n_object = 0
        ob = None

        point_ = "uniform"
        validation_size_ = 0.0
        network_properties_ = {
            "hidden_layers": 4,
            "neurons": 64,
            "residual_parameter": 0.5,
            "kernel_regularizer": 2,
            "regularization_parameter": 0,
            "batch_size": 8192,
            "epochs": 400,
            "activation": "tanh"
        }

        if len_sys_argv == 1:
            retrain_ = 42
            folder_path_ = "models/RayleighOnly_hardBC_4+2x64_400x50_2"
        else:
            retrain_ = int(sys.argv[1]) + 1
            folder_path_ = f"models/HyperparameterSearchHard/IsothermalAtmoAlt_6x64_100x50_seed={sys.argv[1]}"

        shuffle_ = False

    elif len_sys_argv == 17 or len_sys_argv == 13:
        print(sys.argv)
        sampling_seed_ = int(sys.argv[1])
        n_coll_ = int(sys.argv[2])
        n_u_ = int(sys.argv[3])
        n_int_ = int(sys.argv[4])

        n_object = int(sys.argv[5])
        if sys.argv[6] == "None":
            ob = None
        else:
            ob = sys.argv[6]

        folder_path_ = sys.argv[7]
        point_ = sys.argv[8]
        validation_size_ = float(sys.argv[9])
        network_properties_ = json.loads(sys.argv[10])
        retrain_ = sys.argv[11]
        shuffle_ = False if sys.argv[12] == "false" else True
    else:
        raise ValueError("One input is missing")

    return sampling_seed_, n_coll_, n_u_, n_int_, n_object, ob, folder_path_, point_, validation_size_, network_properties_, retrain_, shuffle_


sampling_seed, N_coll, N_u, N_int, N_object, Ob, folder_path, point, validation_size, network_properties, retrain, shuffle = initialize_inputs(len(sys.argv))

if Ec.extrema_values is not None:
    extrema = Ec.extrema_values
    space_dimensions = Ec.space_dimensions
    time_dimension = Ec.time_dimensions
    parameter_dimensions = Ec.parameter_dimensions
    print(space_dimensions, time_dimension, parameter_dimensions)
else:
    print("Using free shape. Make sure you have the functions:")
    print("     - add_boundary(n_samples)")
    print("     - add_collocation(n_samples)")
    print("in the Equation file")
    extrema = None
    space_dimensions = Ec.space_dimensions
    time_dimension = Ec.time_dimensions
try:
    parameters_values = Ec.parameters_values
    parameter_dimensions = parameters_values.shape[0]
    type_point_param = Ec.type_of_points
except AttributeError:
    print("No additional parameter found")
    parameters_values = None
    parameter_dimensions = 0
    type_point_param = None

input_dimensions = Ec.input_dimensions
output_dimension = Ec.output_dimension

print(input_dimensions)
mode = "none"
max_iter = 5000
if network_properties["epochs"] != 1:
    max_iter = 50

if Ob == "cylinder":
    solid_object = ObjectClass.Cylinder(N_object, 1, input_dimensions, time_dimension, extrema, 1, 0, 0)
elif Ob == "square":
    solid_object = ObjectClass.Square(N_object, 1, input_dimensions, time_dimension, extrema, 2, 2, 0, 0)
else:
    solid_object = None

print("######################################")
print("*******Domain Properties********")
print(extrema)
print(input_dimensions)

N_u_train = int(N_u * (1 - validation_size))
N_coll_train = int(N_coll * (1 - validation_size))
N_int_train = int(N_int * (1 - validation_size))
N_object_train = int(N_object * (1 - validation_size))
N_train = N_u_train + N_coll_train + N_int_train + N_object_train

N_u_val = N_u - N_u_train
N_coll_val = N_coll - N_coll_train
N_int_val = N_int - N_int_train
N_object_val = N_object - N_object_train
N_val = N_u_val + N_coll_val + N_int_val + N_object_val

if space_dimensions > 0:
    N_b_train = int(N_u_train / (4 * space_dimensions))
else:
    N_b_train = 0
if time_dimension == 1:
    N_i_train = N_u_train - 2 * space_dimensions * N_b_train
elif time_dimension == 0:
    N_b_train = int(N_u_train / (2 * space_dimensions))
    N_i_train = 0
else:
    raise ValueError()

if space_dimensions > 1:
    N_b_val = int(N_u_val / (4 * space_dimensions))
else:
    N_b_val = 0
if time_dimension == 1:
    N_i_val = N_u_val - 2 * space_dimensions * N_b_val
elif time_dimension == 0:
    N_i_val = 0
else:
    raise ValueError()

print("\n######################################")
print("*******Info Training Points********")
print("Number of train collocation points: ", N_coll_train)
print("Number of initial and boundary points: ", N_u_train, N_i_train, N_b_train)
print("Number of internal points: ", N_int_train)
print("Total number of training points: ", N_train)

print("\n######################################")
print("*******Info Validation Points********")
print("Number of train collocation points: ", N_coll_val)
print("Number of initial and boundary points: ", N_u_val)
print("Number of internal points: ", N_int_val)
print("Total number of training points: ", N_val)

print("\n######################################")
print("*******Network Properties********")
pprint.pprint(network_properties)
batch_dim = network_properties["batch_size"]

print("\n######################################")
print("*******Parameter Dimension********")
print(parameter_dimensions)

if batch_dim == "full":
    batch_dim = N_train

print("DIMENSION")
print(space_dimensions, time_dimension, parameter_dimensions)
training_set_class = DefineDataset(extrema,
                                   parameters_values,
                                   point,
                                   N_coll_train,
                                   N_b_train,
                                   N_i_train,
                                   N_int_train,
                                   batches=batch_dim,
                                   output_dimension=output_dimension,
                                   space_dimensions=space_dimensions,
                                   time_dimensions=time_dimension,
                                   parameter_dimensions=parameter_dimensions,
                                   random_seed=sampling_seed,
                                   obj=solid_object,
                                   shuffle=shuffle,
                                   type_point_param=type_point_param)
training_set_class.assemble_dataset()
training_set_no_batches = training_set_class.data_no_batches

validation_set_class = None
additional_models = None

model = PinnsHardBC(input_dimension=input_dimensions, output_dimension=output_dimension,
                    network_properties=network_properties, additional_models=additional_models)
torch.manual_seed(retrain)
init_xavier(model)
model.num_epochs = network_properties["epochs"]
if torch.cuda.is_available():
    print("Loading model on GPU")
    model.cuda()
if torch.backends.mps.is_available():
    print("Loading model on MPS")
    model.to("mps")

start = time.time()
print("Fitting Model")
model.train()
epoch_ADAM = model.num_epochs

optimizer_LBFGS = optim.LBFGS(model.parameters(), lr=0.8, max_iter=max_iter, max_eval=max_iter, history_size=100,
                              line_search_fn="strong_wolfe",
                              tolerance_change=1.0 * np.finfo(float).eps)
optimizer_ADAM = optim.Adam(model.parameters(), lr=1e-3)
if N_coll_train != 0:
    final_error_train = fit(model, optimizer_ADAM, optimizer_LBFGS, epoch_ADAM, training_set_class, validation_set_clsss=validation_set_class, verbose=True,
                            training_ic=False)
else:
    final_error_train = StandardFit(model, optimizer_ADAM, optimizer_LBFGS, training_set_class, validation_set_clsss=validation_set_class, verbose=True)
end = time.time() - start

print("\nTraining Time: ", end)
print("Final error: ", final_error_train)

model = model.eval()
try:
    final_error_train = float(((10 ** final_error_train) ** 0.5).detach().cpu().numpy())
except AttributeError:
    final_error_train = ((10 ** final_error_train) ** 0.5)
print("\n################################################")
print("Final Training Loss:", final_error_train)
print("################################################")
final_error_val = None
final_error_test = 0

os.mkdir(folder_path)
model_path = folder_path + "/TrainedModel"
os.mkdir(model_path)

torch.save(model, model_path + "/model.pkl")
with open(model_path + os.sep + "Information.csv", "w") as w:
    keys = list(network_properties.keys())
    vals = list(network_properties.values())
    w.write(keys[0])
    for i in range(1, len(keys)):
        w.write("," + keys[i])
    w.write("\n")
    w.write(str(vals[0]))
    for i in range(1, len(vals)):
        w.write("," + str(vals[i]))

with open(folder_path + '/InfoModel.txt', 'w') as file:
    file.write("Nu_train,"
               "Nf_train,"
               "Nint_train,"
               "validation_size,"
               "train_time,"
               "error_train,"
               "error_val,"
               "error_test\n")
    file.write(str(N_u_train) + "," +
               str(N_coll_train) + "," +
               str(N_int_train) + "," +
               str(validation_size) + "," +
               str(end) + "," +
               str(final_error_train) + "," +
               str(final_error_val) + "," +
               str(final_error_test))
