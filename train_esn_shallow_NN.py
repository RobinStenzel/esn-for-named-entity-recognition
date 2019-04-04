import datetime
import errno
# handle output and create directories
import os
import pickle
import sys
from typing import List

import torch
from flair.data import TaggedCorpus, Label
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.training_utils import Metric
from torch import nn, optim
from tqdm import tqdm


def to_device(entity, device):
    if device == "cpu":
        return entity
    else:
        return entity.cuda(device)


def tokenify(sentences):
    token_features = None
    token_targets = None
    for sentence in sentences:
        if token_features is not None:
            token_features = torch.cat((token_features, sentence[0]))
            token_targets = torch.cat((token_targets, sentence[1]))
        else:
            token_features = sentence[0]
            token_targets = sentence[1]

    return token_features, token_targets


def train(model, sentences, batch_size, lr, weight_decay, momentum, epochs, device, log_every):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    
    batches = [sentences[x:x + batch_size] for x in range(0, len(sentences), batch_size)]
    n_batches = len(batches)

    # 1. get the corpus
    corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(NLPTask.GERMEVAL, base_path='resources/tasks')

    # for each epoch
    running_loss = 0.0
    for i in range(epochs):

        # for each batch
        epoch_loss = 0.0
        for j in range(n_batches):
            print("\r Processing epoch:{}/{} and batch: {}/{}".format(i + 1, epochs, j + 1, n_batches), end="")

            # get a batch of sentences
            current_batch = batches[j]

            # tokenify them as word features and targets
            word_features, word_targets = tokenify(current_batch)

            # send to device
            word_features, word_targets = to_device(word_features, device), to_device(word_targets, device)

            # optimizer reset
            optimizer.zero_grad()

            # forward pass
            outputs = model.forward(word_features)

            # calculate loss
            loss = criterion(outputs, word_targets)

            # take a gradient step
            loss.backward()
            optimizer.step()

            epoch_loss += (loss.data.cpu().numpy())

        print(", Loss: {}".format(epoch_loss))
        if i == 0:
            last_dev_f1_score = 0
            best_result = ""
            bad_epochs = 0

        if (i+1) % 10 == 0:
            result = ""
            model = model.eval()
            # let's evaluate the model before training on training set
            predicted_list, _ = evaluate(model, train_sentence_features_targets, tag_dictionary, device, 50,
                                         False)
            train_metric = eval_flair_spans(corpus.train, predicted_list, 32)
            result += "TRAIN: f1-score " + str(train_metric.micro_avg_f_score()) + "- accuracy " + str(train_metric.accuracy()) + "\n"

            predicted_list, _ = evaluate(model, dev_sentence_features_targets, tag_dictionary, device, 50,
                                                 False)
            dev_metric = eval_flair_spans(corpus.dev, predicted_list, 32)

            result += "DEV: f1-score " + str(dev_metric.micro_avg_f_score()) + "- accuracy " + str(dev_metric.accuracy()) + "\n"

            predicted_list, _ = evaluate(model, test_sentence_features_targets, tag_dictionary, device, 50,
                                                 False)
            test_metric = eval_flair_spans(corpus.test, predicted_list, 32)
            result += "TEST: f1-score " + str(test_metric.micro_avg_f_score()) +  "- accuracy " + str(test_metric.accuracy()) + "\n"

            if dev_metric.micro_avg_f_score() > last_dev_f1_score:
                last_dev_f1_score = dev_metric.micro_avg_f_score()
                best_result = result
                print("New best DEV f1-Score {}\n".format(last_dev_f1_score))
                bad_epochs = 0

            else:
                bad_epochs += 1
                if bad_epochs >= 3:
                    print("Too many bad epochs, stopping training")
                    break

            print(result)
            model = model.train()

    return best_result


def evaluate(model, sentences, tag_dictionary, device, batch_size=50, binary=False):
    # batchify again
    batches = [sentences[x:x + batch_size] for x in range(0, len(sentences), batch_size)]
    n_batches = len(batches)

    all_predicted = []
    all_word_targets = []
    for i in tqdm(range(n_batches)):

        # current batch
        current = batches[i]

        # tokenify
        word_features, word_targets = tokenify(current)

        # to device
        word_features, word_targets = to_device(word_features, device), to_device(word_targets, device)

        predicted = model.forward(word_features)

        # take argmax
        predicted = torch.argmax(predicted, dim=1)

        # get the label strings for predicted and target
        predicted = [tag_dictionary.get_item_for_index(int(item)) for item in predicted]
        word_targets = [tag_dictionary.get_item_for_index(int(item)) for item in word_targets]

        if binary:
            predicted = [0 if item == "0" else 1 for item in predicted]
            word_targets = [0 if item == "0" else 1 for item in word_targets]

        # append
        all_predicted.extend(predicted)
        all_word_targets.extend(word_targets)

    # then classification report
    # report = classification_report(all_word_targets, all_predicted)

    return all_predicted, all_word_targets


def eval_flair_spans(data, predicted_list, batch_size, out_path=None):
    metric = Metric('Evaluation')

    mini_batch_size = batch_size
    batches = [data[x:x + mini_batch_size] for x in range(0, len(data), mini_batch_size)]

    lines: List[str] = []
    word_counter = 0
    for batch in batches:
        for sentence in batch:
            for token in sentence.tokens:
                tag = Label(predicted_list[word_counter])
                word_counter += 1
                token.add_tag_label('predicted', tag)

                # append both to file for evaluation
                eval_line = '{} {} {} {}\n'.format(token.text,
                                                   token.get_tag('ner').value, tag.value, tag.score)

                lines.append(eval_line)
            lines.append('\n')

        for sentence in batch:
            # make list of gold tags
            gold_tags = [(tag.tag, str(tag)) for tag in sentence.get_spans('ner')]
            # make list of predicted tags
            predicted_tags = [(tag.tag, str(tag)) for tag in sentence.get_spans('predicted')]

            # check for true positives, false positives and false negatives
            for tag, prediction in predicted_tags:
                if (tag, prediction) in gold_tags:
                    metric.add_tp(tag)
                else:
                    metric.add_fp(tag)

            for tag, gold in gold_tags:
                if (tag, gold) not in predicted_tags:
                    metric.add_fn(tag)
                else:
                    metric.add_tn(tag)

    # add metrics scores at the beginning of the file
    lines.insert(0, str(metric) + "\n\n")

    if out_path is not None:

        # create folder for json and corresponding output
        if not os.path.exists(os.path.dirname(out_path)):
            try:
                os.makedirs(os.path.dirname(out_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(out_path, "w", encoding='utf-8') as outfile:
            outfile.write(''.join(lines))
        #
    # esnWrapper.model.output_activation = output_activation_training
    return metric


# a NN model
class Model(nn.Module):
    def __init__(self, n_inputs, n_outputs, layerConfig, dropout):

        super(Model, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_layers = len(layerConfig)
        self.layerConfig = layerConfig
        self.dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList()

        if len(layerConfig) == 0:  # just a network without any hidden layers
            self.layers.append(nn.Linear(n_inputs, self.n_outputs))
        else:
            self.layers.append(nn.Linear(n_inputs, layerConfig[0]))
            print("Initializing input layer with {} inputs and {} outputs".format(n_inputs, layerConfig[0]))
            lastSize = layerConfig[0]
            for i in range(1, self.n_layers):
                print("Initializing hidden layer {} with {} inputs and {} outputs".format(i, lastSize, layerConfig[i]))

                layer = nn.Linear(lastSize, layerConfig[i])

                # append to the layers
                self.layers.append(layer)
                lastSize = layerConfig[i]
            print("Initializing output layer with {} inputs and {} outputs".format(lastSize, self.n_outputs))
            self.layers.append(nn.Linear(lastSize, self.n_outputs))


    def forward(self, input):
        # pass the input to all layers
        output = input
        for l in range(len(self.layers)):
            layer = self.layers[l]

            output = torch.sigmoid(layer(output)) if l != len(self.layers) - 1 else self.dropout(layer(output))
        return output




# set the device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# load the pre-processed embeddings

if len(sys.argv) > 1:
    save_loc = sys.argv[1]
else:
    save_loc = "please give the embedding location as argument"

train_sentence_features_targets = pickle.load(open(save_loc + "/training_esn_embeddings.p", "rb"))
print("Training embedding loaded")
dev_sentence_features_targets = pickle.load(open(save_loc + "/dev_esn_embeddings.p", "rb"))
print("Develop embedding loaded")
test_sentence_features_targets = pickle.load(open(save_loc + "/testing_esn_embeddings.p", "rb"))
print("Test embedding loaded")
tag_dictionary = pickle.load(open(save_loc + "/tag_dictionary.p", "rb"))

embedding_size = train_sentence_features_targets[0][0].size(1)

# now, we will train a NN model using these embeddings as they already have the context
model = Model(n_inputs=embedding_size, n_outputs=len(tag_dictionary), layerConfig=[1000], dropout=0.5)
model = model.cuda(device)

result = train(model, train_sentence_features_targets, batch_size=32, lr=0.1, weight_decay=1e-5, momentum=0.9,
               epochs=150, device=device, log_every=5)


print(result)
with open(save_loc + "result.txt", "w", encoding='utf-8') as outfile:
    outfile.write(result)
