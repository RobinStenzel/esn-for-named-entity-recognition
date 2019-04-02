import os
import sys
from typing import List

from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher, NLPTask
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings

sys.path.insert(1, os.path.join(sys.path[0], '..'))

# 1. get the corpus
corpus: TaggedCorpus = NLPTaskDataFetcher.load_corpus(NLPTask.GERMEVAL, base_path='resources/tasks')
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('de'),
    #uncomment to use Flair embeddings
    #FlairEmbeddings('german-forward'),
    #FlairEmbeddings('german-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)


# initialize sequence tagger
from flair.models import SequenceTagger
print("embeddings length {}".format(embeddings.embedding_length))

tagger: SequenceTagger = SequenceTagger(hidden_size=2048,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=False,
                                        use_rnn=True)

# initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('resources/taggers/example-ner',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150,
              embeddings_in_memory=True)