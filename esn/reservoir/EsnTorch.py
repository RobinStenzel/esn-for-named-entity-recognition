import torch
import torch.autograd as autograd
import numpy as np
from torch import optim
from esn.reservoir import ActivationFunctions, ReservoirTopology as topology
from flair.data import Label
from tqdm import tqdm


def to_device(item, device):
    if device == "cpu":
        return item
    else:
        return item.cuda(device)


class EsnTorch(torch.nn.Module):
    def __init__(self,
                 parameters,
                 input_weight,
                 reservoir_weight,
                 output_weight_init,
                 device,
                 reservoir_activation_function=ActivationFunctions.HyperbolicTangent(),
                 output_activation_function=ActivationFunctions.Linear(), ):
        super(EsnTorch, self).__init__()

        # initializations
        self.device = device
        self.param = parameters
        self.W_i = autograd.Variable(to_device(input_weight, device), requires_grad=False)
        self.W_r = autograd.Variable(to_device(reservoir_weight, device), requires_grad=False)
        self.reservoir_activation = reservoir_activation_function
        self.output_activation = output_activation_function
        self.r_t = to_device(torch.zeros((self.param["size"], 1)).type(torch.float32), device)

        # model parameter
        self.W_o = torch.nn.Parameter(to_device(output_weight_init, device), requires_grad=True)

        # force spectral radius
        # Make the reservoir weight matrix - a unit spectral radius
        rad = torch.max(torch.abs(torch.eig(reservoir_weight)[0]))
        self.W_r = self.W_r / rad

        # Force spectral radius
        self.W_r = self.W_r * self.param["spectral_radius"]

    def forward(self, input):
        # reservoir activation
        input = input.view(input.shape[0], -1)
        term1 = self.W_i.mm(input)
        term2 = self.W_r.mm(self.r_t)
        self.r_t = (1.0 - self.param["leaking_rate"]) * self.r_t + self.param[
            "leaking_rate"] * self.reservoir_activation(term1 + term2)

        # output activation
        output = torch.mm(self.W_o, self.r_t)
        output = self.output_activation(output)
        return output, self.r_t

    def reset_states(self):
        self.r_t = to_device(torch.zeros((self.param["size"], 1)).type(torch.float32), self.device)


class EsnWrapper():
    def __init__(self, parameters, input_dim, output_dim, reservoir_activation, output_activation, device):
        self.parameters = parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.bidirectional = parameters["bidirectional"]
        self.use_pooling = parameters["pooling_hidden_states"]
        if self.use_pooling:
            self.max_pool = torch.nn.MaxPool1d(2, stride=2)

        self.use_kernel_trick = self.parameters["jaeger_kernel_trick"]


        # initialize model
        input_weight = torch.from_numpy(topology.RandomInputTopology(inputSize=input_dim,
                                                                     reservoirSize=parameters["size"],
                                                                     inputConnectivity=parameters[
                                                                         "input_connectivity"]).generateWeightMatrix(
            scaling=parameters["input_scaling"]))
        reservoir_weight = torch.from_numpy(topology.RandomReservoirTopology(size=parameters["size"],
                                                                             connectivity=parameters[
                                                                                 "reservoir_connectivity"]).generateWeightMatrix(
            scaling=parameters["reservoir_scaling"]))

        output_weight = torch.randn((output_dim, parameters["size"])).double()

        self.model = EsnTorch(parameters=parameters,
                              input_weight=input_weight.type(torch.float32),
                              reservoir_weight=reservoir_weight.type(torch.float32),
                              output_weight_init=output_weight.type(torch.float32),
                              reservoir_activation_function=reservoir_activation,
                              output_activation_function=output_activation,
                              device=device)

        # loss and optimizer
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.model.parameters())

    def train(self, sentences):

        # for each sentence
        loss_batch = 0
        for sentence in sentences:

            # split into inputs and targets
            token_embeddings, token_targets = sentence

            # reset states
            self.model.reset_states()

            # reset gradients
            self.optimizer.zero_grad()

            # for each token
            for j in range(len(token_embeddings)):
                # variables for input and output
                input = to_device(token_embeddings[j].type(torch.float32), self.device)
                target = to_device(torch.tensor(token_targets[j]).view(1, -1), self.device)
                target = torch.squeeze(target, dim=0)

                # forward pass
                output, _ = self.model(input)
                output = output.view(1, -1)

                # compute loss
                loss = self.loss_function(output, target)

                # backward pass
                if j > self.parameters["initial_transient"]:
                    loss.backward()
                    loss_batch += (loss.data.cpu().numpy() / len(token_embeddings))

            self.optimizer.step()

        return loss_batch / len(sentences)

    # receives list of sentences with already embedded tokens, returns hidden state for sentence
    def embed_sentence(self, train_sentence_features_targets):

        esn_embedded_train_sentence_features_targets = []

        # for each sentence
        for i, sentence_features in tqdm(enumerate(train_sentence_features_targets)):

            sentence_tensor, tag_tensor = sentence_features

            embedded_sentence_tensor = []
            bi_embedded_sentence_tensor = []

            # to device
            sentence_tensor = to_device(sentence_tensor, self.device)

            # reset states
            self.model.reset_states()

            # for each token (forward)
            for j in range(len(sentence_tensor)):
                # variables for input and output
                input = sentence_tensor[j].type(torch.float32)
                washout_number = self.parameters["initial_transient"]*len(sentence_tensor)
                if j >= washout_number:
                    # forward pass
                    output, contextual_word_embedding = self.model(input)
                    forward_contextual_word_embedding = torch.squeeze(contextual_word_embedding.view(1, -1), dim=0)
                else:
                    forward_contextual_word_embedding = torch.zeros(self.parameters["size"], dtype=torch.float32)

                if self.use_pooling:
                    reshaped_forward = forward_contextual_word_embedding.view(1, 1, -1)
                    pooled_forward = self.max_pool(reshaped_forward)
                    forward_contextual_word_embedding = pooled_forward.view(-1)

                if self.use_kernel_trick:
                    forward_contextual_word_embedding = torch.cat(
                        (forward_contextual_word_embedding,
                         torch.mul(forward_contextual_word_embedding, forward_contextual_word_embedding)), 0)

                embedded_sentence_tensor.append(forward_contextual_word_embedding)

            if not self.bidirectional:
                esn_embedded_train_sentence_features_targets.append((torch.stack(embedded_sentence_tensor), tag_tensor))
            else:

                # for each token (backward)
                # reset states
                self.model.reset_states()

                for k in range(len(sentence_tensor)):
                    # variables for input and output
                    input = sentence_tensor[len(sentence_tensor)-1 - k].type(torch.float32)
                    washout_number = self.parameters["initial_transient"]*len(sentence_tensor)

                    if k >= washout_number:
                        # forward pass
                        output, contextual_word_embedding = self.model(input)
                        backward_contextual_word_embedding = torch.squeeze(contextual_word_embedding.view(1, -1), dim=0)
                    else:
                        backward_contextual_word_embedding = torch.zeros(self.parameters["size"], dtype=torch.float32)

                    forward_contextual_word_embedding = embedded_sentence_tensor[len(sentence_tensor)-1 - k]

                    if self.use_pooling:
                        reshaped_backward = backward_contextual_word_embedding.view(1, 1, -1)
                        pooled_backward = self.max_pool(reshaped_backward)
                        backward_contextual_word_embedding = pooled_backward.view(-1)

                    if self.use_kernel_trick:
                        backward_contextual_word_embedding = torch.cat(
                            (backward_contextual_word_embedding,
                             torch.mul(backward_contextual_word_embedding, backward_contextual_word_embedding)), 0)




                    bidirectional_embedding = torch.cat(
                        (forward_contextual_word_embedding, backward_contextual_word_embedding), 0)
                    bi_embedded_sentence_tensor.insert(0, bidirectional_embedding)

                esn_embedded_train_sentence_features_targets.append(
                    (torch.stack(bi_embedded_sentence_tensor), tag_tensor))

        return esn_embedded_train_sentence_features_targets

    def eval(self, sentences):
        # for each sentence
        batch_predicted_list = []
        batch_gold_list = []
        for sentence in sentences:

            # split into inputs and targets
            token_embeddings, token_targets = sentence

            # for each token
            for j in range(len(token_embeddings)):
                # variables for input and output
                input = token_embeddings[j].type(torch.float32)

                # forward pass
                output = self.model(input)
                max_score, predicted_tag = output.max(0)
                gold_tag = token_targets[j]

                batch_predicted_list.append(predicted_tag)
                batch_gold_list.append(gold_tag)

        return batch_gold_list, batch_predicted_list

    def evalFlair(self, sentences, tag_dictionary):
        # for each sentence
        tags = []
        loss_batch = 0
        for sentence in sentences:
            tag_seq = []
            confidences = []

            # reset states
            self.model.reset_states()

            # reset gradients
            self.optimizer.zero_grad()

            # split into inputs and targets
            token_embeddings, token_targets = sentence

            # for each token
            for j in range(len(token_embeddings)):
                # variables for input and output
                input = token_embeddings[j].type(torch.float32)
                target = torch.tensor(token_targets[j]).view(1, -1)
                target = torch.squeeze(target, dim=0)

                # forward pass
                output = self.model(input)
                max_score, predicted_tag = output.max(0)

                output = output.view(1, -1)

                # compute loss
                loss = self.loss_function(output, target)

                # backward pass
                loss.backward()
                loss_batch += (loss.data.numpy() / len(token_embeddings))

                tag_seq.append(predicted_tag)
                confidences.append(round(max_score[0].item(), 2))

            tags.append([Label(tag_dictionary.get_item_for_index(tag), conf)
                         for tag, conf in zip(tag_seq, confidences)])

        return tags, loss_batch / len(sentences)

    def predict(self, input):
        output = self.model.forward(input)
        return output
