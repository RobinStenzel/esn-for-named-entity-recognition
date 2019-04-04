# read in parameters from json
import json
# handle output and create directories
import os
import pickle
# to access arguments
import sys
from typing import List

import torch
# use flair to read in data and for generating flair embeddings
from flair.data import TaggedCorpus, Sentence
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
# plots
from tqdm import tqdm

from esn.reservoir import EsnTorch as ESN, ActivationFunctions as act

embeddings_in_memory = False


def clear_embeddings(sentences: List[Sentence], also_clear_word_embeddings=not embeddings_in_memory):
    """
    Clears the embeddings from all given sentences.
    :param sentences: list of sentences
    """
    for sentence in sentences:
        sentence.clear_embeddings(also_clear_word_embeddings=also_clear_word_embeddings)


def prepare_batch_as_lists(sentences: List[Sentence], pre_trained_embeddings):
    # embed batch in the trained embedding
    pre_trained_embeddings.embed(sentences)

    # list of sentences and their tags
    sentences_and_tags = []

    for s_id, sentence in enumerate(sentences):

        # get token embeddings and their tags
        all_token_embeddings = []
        all_token_tags = []
        for token in sentence:
            token_embedding = token.get_embedding()
            token_tag = tag_dictionary.get_idx_for_item(token.get_tag(tag_type).value)
            all_token_embeddings.append(token_embedding)
            all_token_tags.append(token_tag)

        # add it to the sentences
        sentences_and_tags.append((all_token_embeddings, all_token_tags))

    embeddings_in_memory = True

    return sentences_and_tags


# transforms batch (list of sentence) into sentence tensor (batch x seq_len x embedding) and tag tensor (batch x seq_len x tagset_size)
def prepare_batch(sentences: List[Sentence], pre_trained_embeddings, sort=True):
    tag_type = 'ner'

    # embed batch in the trained embedding
    pre_trained_embeddings.embed(sentences)

    # if sorting is enabled, sort sentences by number of tokens
    if sort:
        sentences.sort(key=lambda x: len(x), reverse=True)

    lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
    tag_list: List = []

    longest_token_sequence_in_batch: int = lengths[0]

    # initialize zero-padded word embeddings tensor
    sentence_tensor = torch.zeros([len(sentences),
                                   longest_token_sequence_in_batch,
                                   pre_trained_embeddings.embedding_length],
                                  dtype=torch.float)

    for s_id, sentence in enumerate(sentences):
        # fill values with word embeddings
        sentence_tensor[s_id][:len(sentence)] = torch.cat([token.get_embedding().unsqueeze(0)
                                                           for token in sentence], 0)
    return sentence_tensor, lengths


def getFeaturesAndTargets(list_of_sentences, embeddings, tag_dictionary):
    # feature and target
    sentence_features_and_targets = []
    n_sentences = len(list_of_sentences)
    for i in tqdm(range(n_sentences)):
        # current sentence
        sentence = list_of_sentences[i]

        # get embeddings for sentence
        embeddings.embed(sentence)

        # get token level embedding features
        token_level_features = torch.cat([token.get_embedding().unsqueeze(0) for token in sentence], 0)

        # get token level targets
        token_level_targets = torch.tensor(
            [tag_dictionary.get_idx_for_item(token.get_tag("ner").value) for token in sentence])

        # clear the sentence embedding
        sentence.clear_embeddings()

        sentence_features_and_targets.append((token_level_features, token_level_targets))
    return sentence_features_and_targets


def save_embeddings_in_json_folder(json_path: object, tag_dictionary: object, train: object, dev: object,
                                   test: object) -> object:
    save_loc = "saved_esn_embeddings/{}".format(json_path[:-5])

    # create folder for json and corresponding output
    try:
        os.makedirs(save_loc)
    except FileExistsError:
        # directory already exists
        pass

    pickle.dump(tag_dictionary, open(save_loc + "/tag_dictionary.p", "wb"))
    pickle.dump(train, open(save_loc + "/training_esn_embeddings.p", "wb"))
    pickle.dump(dev, open(save_loc + "/dev_esn_embeddings.p", "wb"))
    pickle.dump(test, open(save_loc + "/testing_esn_embeddings.p", "wb"))

    with open(save_loc + "/parameters.json", 'w') as fp:
        json.dump(parameters, fp)


# read in parameters as json file from 2nd argument
json_path = sys.argv[1]
parameters = json.load(open(json_path))

# 1. get the corpus
corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(NLPTask.GERMEVAL, base_path='resources/tasks/')

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('de'),
    # uncomment to test Flair Embeddings
    #FlairEmbeddings('german-forward'),
    #FlairEmbeddings('german-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

train_data = corpus.train
dev_data = corpus.dev
test_data = corpus.test

mini_batch_size = parameters["batch_size"]

totalNumberBatches = len(corpus.train) / mini_batch_size

batches = [train_data[x:x + mini_batch_size] for x in range(0, len(train_data), mini_batch_size)]

# Train an echo state network

parameters["bidirectional"] = True
parameters["pooling_hidden_states"] = False
parameters["jaeger_kernel_trick"] = False

# select the device based on availability
useGPU = True
device = "cuda:0" if torch.cuda.is_available() and useGPU else "cpu"

res = ESN.EsnWrapper(parameters=parameters,
                     input_dim=embeddings.embedding_length,
                     output_dim=len(tag_dictionary),
                     reservoir_activation=act.HyperbolicTangent(),
                     output_activation=act.Linear(),
                     device=device)

number_of_epochs = parameters["epochs"]

print("Preprocess data corpus for train, dev and test")
train_sentence_features_targets = getFeaturesAndTargets(corpus.train, embeddings, tag_dictionary)
dev_sentence_features_targets = getFeaturesAndTargets(corpus.dev, embeddings, tag_dictionary)
test_sentence_features_targets = getFeaturesAndTargets(corpus.test, embeddings, tag_dictionary)

print("Create contextual word embedding for train, dev and test")
train_sentence_features_targets = res.embed_sentence(train_sentence_features_targets)
dev_sentence_features_targets = res.embed_sentence(dev_sentence_features_targets)
test_sentence_features_targets = res.embed_sentence(test_sentence_features_targets)

save_embeddings_in_json_folder(json_path, tag_dictionary, train_sentence_features_targets,
                               dev_sentence_features_targets, test_sentence_features_targets)
