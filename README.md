Using Echo State Networks for Named Entity Recognition

## Instructions to reproduce the results described in the paper

### Generate Echo State Embeddings

Execute the script generate_esn_embedding.py with json file as second argument.
One json file with the tuned parameters can be found in the jsons foulder. 
It should look like this:

python generate_esn_embedding.py jsons/tuned_parameters.json

The embedding will be saved under saved_esn_embeddings/*json name*/.

The embeddings can be generated for FastText Word Embedding or for Flair Word Embedding.
For this purpose, uncomment the desired embeddings in the script.

```python
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('de'),
    # uncomment to test Flair Embeddings
    FlairEmbeddings('german-forward'),
    FlairEmbeddings('german-backward'),
]
```


### Evaluate Embeddings with Logisitic Regression or a shallow Neural Network

Execute either the train_esn_logreg.py or train_esn_shallow_NN.py with the folder of the esn embedding as second argument:

train_esn_logreg.py saved_esn_embeddings/*json name*/

### Reproduce results for the LSTM, CRF and Logistic Regression

Go into the flair folder and execute the trainGermEval.py.
Within the script you can again change the word embedding aswell as if CRF or LSTM are used. If both are set to false, logistic regression will be used. Besides, you can try different hidden_sizes. If the LSTM is used the embedding will be of size 2*hidden_size.
```python
tagger: SequenceTagger = SequenceTagger(hidden_size=2048,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=False,
                                        use_rnn=True)

```

