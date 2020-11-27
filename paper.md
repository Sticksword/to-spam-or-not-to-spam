# No phishing allowed: Detecting spear phishing attacks using deep learning

## 1. Introduction

Phishing is the fraudulent attempt to steal sensitive information or data, such as usernames, passwords and credit card details, by disguising oneself as a trustworthy entity in any form of electronic communication. These attacks occur in all industries and according to Verizon's yearly report, 32% of breaches involve phishing and 33% of breaches included social attacks. [1]

The focus of this project will be on email phishing attacks. Symantec released a security threat report stating that 65% of attacker groups used spear phishing as the primary infection vector. [2] Our aim is to catch these spear phishing emails.

## 2 Current landscape

## 3 Approach

The classification task is to identify whether or not an email is phishy or ham.

### 3.1 Feature Identification

One key idea behind text classification is document representation. Our initial approach includes a simple CountVectorizer and a TF-IDF Vectorizer. We process an email document and pass the counts/scores of the vocab into our classifier.

We can take this a step further and represent our document as a series of "word vectors" that embed semantic meaning (using some probabilistic technique). Creating word vectors is the process of taking a large corpus of text and creating a vector for each word such that words that share common contexts in the corpus are located in close proximity to one another in the vector space.

### 3.2 Traditional ML vs Deep Learning

## 4 Design & Implementation

### 4.1 Email Text Preprocessing

### 4.2 Tokenizing the Text

### 4.3 Traditional Random Forest and Logistic Regression Classifiers

### 4.4 Recurrent Neural Network Classifier

### 4.5 Input sequence embedding

### 4.6 Cutout Pruning

## 5 Evaluation

### 5.1 Training

### 5.2 Evaluation Metrics

## 6 Conclusion

## References

[[1] 2019 Verizon Data Breach Investigations Report](https://www.nist.gov/system/files/documents/2019/10/16/1-2-dbir-widup.pdf)
[[2] 2019 Symantec Internet Security Threat Report (ISRT)](https://docs.broadcom.com/doc/istr-24-2019-en)
[Catching the Phish](https://arxiv.org/pdf/1908.03640.pdf)
[Spam Filtering with Naive Bayes](http://www2.aueb.gr/users/ion/docs/ceas2006_paper.pdf)
[Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)
[Improved Phishing Detection using Model-Based Features](https://www.ceas.cc/2008/papers/ceas2008-paper-44.pdf)
