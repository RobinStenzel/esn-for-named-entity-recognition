from reservoir import EsnTorch as ESN, ActivationFunctions as act
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
        query = np.array(query, dtype=np.float64).reshape(2,1)

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
parameters = {}
parameters["size"] = 500
parameters["initial_transient"] = 50
parameters["reg_factor"] = 1e-4
parameters["leaking_rate"] = 0.3
parameters["spectral_radius"] = 0.79
parameters["input_connectivity"] = 0.5
parameters["reservoir_connectivity"] = 0.5
parameters["input_scaling"] = 0.5
parameters["reservoir_scaling"] = 0.5

# Plot variables to hold data for comparison
plot_names = []
predicted = []
plot_names.append("actual")
predicted.append(testing_data)

# Train recursive min
res = ESN.EsnWrapper(parameters=parameters,
                     input_data=torch.from_numpy(input_training),
                     output_data=torch.from_numpy(output_training),
                     reservoir_activation=act.HyperbolicTangent(),
                     output_activation=act.Linear())
res.train(n_iter=5)

# Warm up
warmup_points = input_training[-parameters["initial_transient"]:, :].T
for i in range(warmup_points.shape[1]):
    warmup_output = res.predict(torch.from_numpy(warmup_points[:,i]))

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



