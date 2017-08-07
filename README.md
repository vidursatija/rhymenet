# RhymeNet

### TL;DR
Program generates rap using LSTMs

## Model
Each word has 2 embeddings. 1 is the semantic embedding and the other is the rhyme embedding which is extracted from the nltk cmu dictionary. The model then learns from rap lyrics and predicts paragraphs.

## Dependencies
1. NLTK
2. Tensorflow
3. Numpy
4. [lyrics_scraper](https://github.com/rohankshir/lyrics_scraper)
