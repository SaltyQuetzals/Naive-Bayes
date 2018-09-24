# Naive-Bayes

An implementation of a Naive Bayes' classifier in Python.

## Execution

To run the implemented Naive Bayes model, simply `cd` into the `python` directory, and run the following (assuming your default Python version is 2.7)

```language=bash
python NaiveBayes.py [-f] [-b] ../data/imdb1/
```

There are two optional parameters that can be provided to the application.

- `-f` - Filters out stopwords from the corpus
- `-b` - Caps word counts in each document at 1 (indicating the word is present)

## Analysis

See `analysis.pdf` for the analysis of the application.