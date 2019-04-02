from reservoir import Esn as ESN, ReservoirTopology as topology
import numpy as np
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch


def split_into_training_and_testing(data, horizon):
    training = data[:data.shape[0]-horizon]
    testing = data[data.shape[0]-horizon:]
    return training, testing


def form_feature_vectors(data):
    size = data.shape[0]
    input = np.hstack((np.ones((size-1, 1)),data[:size-1].reshape((size-1, 1))))
    output = data[1:size].reshape((size-1, 1))
    return input, output


def predict_future(network, seed, horizon):
    # Predict future values
    predicted = []
    last_available = seed
    for i in range(horizon):
        query = [1.0]
        query.append(last_available)
        query = np.array(query).reshape(2,1)

        # Predict the next point
        next = network.predict(torch.from_numpy(query))
        predicted.append(next)

        last_available = next

    predicted = np.array(predicted).reshape((horizon, 1))
    return predicted

# Read data from the file
data = np.loadtxt('MackeyGlass_t17.txt')

# Input data
n = 5000
orig_data = data[:n].reshape((n, 1))


# Split training and testing
horizon = 300
training_data, testing_data = split_into_training_and_testing(data, horizon)

# Form feature vectors
input_training, output_training = form_feature_vectors(training_data)

# Train an echo state network
size = 500
initial_transient = 50
reg_factor = 1e-4
leaking_rate = 0.3
spectral_radius = 0.79

# Input-to-reservoir fully connected
input_weight = topology.ClassicInputTopology(inputSize=input_training.shape[1], reservoirSize=size).generateWeightMatrix(scaling=1.0)

# Reservoir-to-reservoir fully connected
reservoir_weight = topology.ClassicReservoirTopology(size=size).generateWeightMatrix(scaling=1.0)


# Plot variables to hold data for comparison
plot_names = []
predicted = []
plot_names.append("actual")
predicted.append(testing_data)

# Train full batch - classic closed form solution
res = ESN.Reservoir(size=size,
                    input_data=torch.from_numpy(input_training.T),
                    output_data=torch.from_numpy(output_training.T),
                    leaking_rate=leaking_rate,
                    spectral_radius=spectral_radius,
                    input_weight=torch.from_numpy(input_weight),
                    reservoir_weight=torch.from_numpy(reservoir_weight),
                    initial_transient=initial_transient,
                    reg_factor=reg_factor)
res.train_full_batch()

# Warm up
warmup_output = res.predict(torch.from_numpy(input_training[-initial_transient:, :].T))

# Predict future values
predicted_testing_data_recursive = predict_future(res, training_data[-1], horizon)
predicted.append(predicted_testing_data_recursive)
plot_names.append("pred")


# Plot everything now and save it!
fig = plt.figure(num=1, facecolor="white")
gs = gridspec.GridSpec(nrows=2, ncols=1)
sp = plt.subplot(gs[0])
for i in range(len(plot_names)):
    sp.plot(np.arange(predicted[i].shape[0]), predicted[i], label=plot_names[i])
sp.set_xlabel('t')
sp.set_ylabel('y')
legend = sp.legend(loc="lower right", fontsize=5)
plt.savefig('outputs/predicition_comparison.pdf')



