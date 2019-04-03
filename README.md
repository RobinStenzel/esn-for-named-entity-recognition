# Echo State Networks for Named Entity Recognition

## Instructions to reproduce the results described in the paper

### Generate Echo State Embeddings

Execute the generate esn embedding script and pass a json as an argument. There is one example json file with our tuned parameters in the json folder.
Within the json, you can change the hyperparameters of the ESN like reservoir size, leaky rate etc.

The embeddings can be generated for FastText Word Embedding or for Flair Word Embedding.
For this purpose, uncomment the desired embeddings in the script.

```python
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('de'),
    # uncomment to test Flair Embeddings
    # FlairEmbeddings('german-forward'),
    # FlairEmbeddings('german-backward'),
]
```

Then execute the generation:

```console
console:~/esn-for-named-entity-recognition$ python3 generate_esn_embedding.py jsons/tuned_parameters.json
```

The embedding for training, develop and test will be saved in saved_esn_embeddings/jsons/tuned_esn_parameters/.

### Evaluate Embeddings with Logisitic Regression or a shallow Neural Network

Run either logistic regression or a shallow neural network on the computed embeddings.

```console
console:~/esn-for-named-entity-recognition$ python3 train_esn_logreg.py saved_esn_embeddings/jsons/tuned_esn_parameter

```

```console
console:~/esn-for-named-entity-recognition$ python3 train_esn_shallow_NN.py saved_esn_embeddings/jsons/tuned_esn_parameter

```
This will output a evalutation of the test and develop set every 10 epochs and stop after 150 epochs.

### Reproduce results for the LSTM, CRF and Logistic Regression

Go into the flair folder and execute the trainGermEval.py.

```console
console:~/esn-for-named-entity-recognition/flair$ spython trainGermEval.py 

```

Within the script you can again change the word embedding like mentioned before. You can also experiment with different combinations of CRF and LSTM. If both are set to false, logistic regression will be used. Besides, you can try different hidden_sizes. If the LSTM is used the contextual embedding will be of size 2 times the hidden_size.

```python
tagger: SequenceTagger = SequenceTagger(hidden_size=2048,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=False,
                                        use_rnn=True)

```

