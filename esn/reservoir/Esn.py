import torch
from reservoir import ActivationFunctions

class Reservoir:
    def __init__(self,
                 size,
                 spectral_radius,
                 leaking_rate,
                 initial_transient,
                 input_data,
                 output_data,
                 input_weight=None,
                 reservoir_weight=None,
                 reservoir_activation_function=ActivationFunctions.HyperbolicTangent(),
                 output_activation_function=ActivationFunctions.Linear(),
                 reg_factor=1e-10):
        self.n_r = size
        self.spectral_radius = spectral_radius
        self.leaking_rate = leaking_rate
        self.initial_transient = initial_transient
        self.X = input_data
        self.Y = output_data

        # Initialize weights
        self.input_d, self.input_t = self.X.shape
        self.output_d, self.output_t = self.Y.shape
        self.n_i = self.input_d
        self.n_o = self.output_d
        self.W_i = input_weight
        self.W_r = reservoir_weight
        self.W_o = torch.zeros((self.n_o, self.n_r)).double()

        # force spectral radius
        self.__force_spectral_radius()

        # Internal states
        self.latest_r = torch.zeros((self.n_r, 1)).double()

        # Activation functions
        self.reservoir_activation = reservoir_activation_function
        self.output_activation = output_activation_function

        # Reg factor
        self.reg_factor = reg_factor

    def __force_spectral_radius(self):
        # Make the reservoir weight matrix - a unit spectral radius
        rad = torch.max(torch.abs(torch.eig(self.W_r)[0]))
        self.W_r = self.W_r / rad

        # Force spectral radius
        self.W_r = self.W_r * self.spectral_radius

    def collect_internal_states(self):
        T = self.X.shape[1]
        R = torch.zeros((self.n_r, T)).double()
        range_ = range(T)
        for t in range_:
            state = self.get_internal_state(self.X[:,t].view((self.n_i, -1)))
            if t > self.initial_transient:
                R[:, t] = state.flatten()
        return R[:, self.initial_transient:]

    def pseduo_inverse(self, X):
        X_transpose = torch.t(X)
        covinv = torch.inverse(X.mm(X_transpose) + self.reg_factor * torch.eye(X.shape[0]).double())
        pinv = X_transpose.mm(covinv)
        return pinv

    def train_full_batch(self):
        # Collect internal states
        self.R = self.collect_internal_states()

        # Compute pseudo inverse
        pseudo = self.pseduo_inverse(self.R)

        # Compute output weight
        self.W_o = torch.mm(self.Y[:, self.initial_transient:], pseudo)

        # reset the internal states after training
        self.reset_internal_states()

    def get_internal_state(self, input):
        # get the latest state
        r_t = self.latest_r

        term1 = self.W_i.mm(input)
        term2 = self.W_r.mm(r_t)
        r_t = (1.0 - self.leaking_rate) * r_t + self.leaking_rate * self.reservoir_activation(term1 + term2)

        # update the latest state
        self.latest_r = r_t

        return r_t

    def predict(self, test_input):
        _, test_input_t = test_input.shape
        r_t = self.latest_r
        test_output = torch.zeros((self.output_d, test_input_t)).double()

        for t in range(test_input_t):
            # reservoir activation
            r_t = self.get_internal_state(test_input[:, t].view(self.n_i, 1))

            # output
            output = self.output_activation(self.W_o.mm(r_t))
            test_output[:, t] = output

        return test_output.numpy()

    def reset_internal_states(self):
        self.latest_r = torch.zeros((self.n_r,1)).double()