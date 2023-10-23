# spacy-stemmer

Stemmer component for [spaCy](https://spacy.io/) using the [NLTK](https://www.nltk.org/) [SnowballStemmer](https://www.nltk.org/api/nltk.stem.snowball.html?highlight=stem#nltk.stem.snowball.SnowballStemmer).

Note that stemming is [not recommended](https://github.com/explosion/spaCy/issues/327), but this pipeline component can be useful for replicating the behaviour of other systems.


## Installation

Use `pip` to install from GitHub:
```console
pip install git+https://github.com/pdhall99/spacy-stemmer.git
```


## Usage

The stemmer component sets the token attribute `._.stem` to be its stemmed form:

```python
>>> import spacy
>>> nlp = spacy.blank("en")
>>> nlp.add_pipe("stemmer")
>>> doc = nlp("Absolute power corrupts absolutely")
>>> for token in doc:
...     print(token._.stem)
absolut
power
corrupt
absolut
```
