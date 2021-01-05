# No Phishing Allowed: Detecting Spear Phishing Attacks using Deep Learning

## [Paper](paper.md)

## Pre-reads

https://stackoverflow.com/questions/10059594/a-simple-explanation-of-naive-bayes-classification?rq=1
https://stackoverflow.com/questions/37405617/how-to-use-tf-idf-with-naive-bayes
https://stats.stackexchange.com/questions/58564/help-me-understand-bayesian-prior-and-posterior-distributions

One key idea behind text classification is document representation. Our initial approach includes a simple CountVectorizer and a TF-IDF Vectorizer. We process an email document and pass the counts/scores of the vocab into our classifier.

We can take this a step further and represent our document as a series of "word vectors" that embed semantic meaning (using some probabilistic technique). Creating word vectors is the process of taking a large corpus of text and creating a vector for each word such that words that share common contexts in the corpus are located in close proximity to one another in the vector space.

A simple approach would be to pass in each word as an integer representing the `i_th_` most common word in the vocab (+/- stop words). We would then add an Embedding Layer that learns the word vector for each word in the vocabulary. One approach is the Skip Gram model which basically tries to predict context words given center word. The word vector that is produced for each word represents the probability a word in the vocab can be found next to the given word. Going back to the original example, an email document comprised of words would be converted into a series of integers representing a word, and then converted into a word vector from that integer.

One caveat to note is that word2vec/doc2vec can perform worse than a traditional count/if-idf vectorizer because of a small training corpus.

## terminology

* batch = a chunk of the entire dataset
* epoch = when an ENTIRE dataset is passed forward and backward through the neural network only ONCE
* iterations = number of batches needed to complete one epoch

Thus the number of batches is equal to number of iterations for one epoch.

Let’s say we have 2000 training examples that we are going to use. If we divide the dataset of 2000 examples into batches of 500, then it will take 4 iterations to complete 1 epoch.

## Setup

You will need `pipenv` to install the dependencies.

We are not using [UCI's 1999 Spambase](http://archive.ics.uci.edu/ml/datasets/Spambase/) because it's too dated and also too preprocessed.

We are going to use many different sources:

* enron1 from [the enron dataset](http://www2.aueb.gr/users/ion/data/enron-spam/)
  * located in `enron1`
  * [associated paper](http://www2.aueb.gr/users/ion/docs/ceas2006_paper.pdf)
* [spam assassin's 2003 and 2005 public corpus](http://spamassassin.apache.org/old/publiccorpus/)
  * first trained on `easy_ham` and `spam`
  * later can try `spam_2` and `hard_ham`
* email marked as spam sent to my inbox, first as a nice mbox file, then converted to text
  * use Google Takeout
  * convert using `mbox-to-txt.py` script
  * stored in `gmail_spam_examples`
  * each data file contains subject and body of an email
* Square Marketing spam data

## Outputs

Currently exporting xgboost models in their model IO bin format and any other sklearn model in PMML format using [sklearn2pmml](https://github.com/jpmml/sklearn2pmml). [Decent article on sklearn2pmml](https://medium.com/@xiaowei_6531/putting-sci-kit-learn-models-into-production-with-pmml-1d17b5fc8123).

* [stackoverflow question/example](https://stackoverflow.com/questions/44560823/generate-pmml-for-text-classification-pipeline-in-python)
* [converting sklearn models into pmml sometimes takes days if you don't bound the number of features](https://github.com/jpmml/sklearn2pmml/issues/88)
* [also cannot use l2 normalization at the time of this issue report](https://github.com/jpmml/jpmml-sklearn/issues/28)
