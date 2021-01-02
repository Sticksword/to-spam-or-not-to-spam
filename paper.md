# No Phishing Allowed: Detecting Spear Phishing Attacks using Deep Learning

## Keywords

* Phishing
* Machine Learning
* Recurrent Neural Networks
* Attention
* Natural Language Processing
* Web security.

## 1 Introduction

Phishing is the fraudulent attempt to steal sensitive information such as usernames, passwords, and credit card details, by disguising oneself as a trustworthy entity in any form of electronic communication. These attacks occur in all industries and according to Verizon's 2019 annual report, 32% of breaches involve phishing and 33% of breaches included social attacks. [1]

The focus of this project will be on email phishing attacks. Symantec released a security threat report stating that 65% of attacker groups used spear phishing as the primary infection vector. [2] The dynamic nature of phishing, with new trends and challenges constantly emerging, motivates a more adaptive filtering approach. Machine learning (ML) has become the de-facto standard for classification purposes over many fields, email classification included. Developing an ML-based classifier to enable phishing filtering is the approach we investigate in this paper. Our classifier analyses the text of the email and, in particular, the email’s language structure. Our aim is to catch these spear phishing emails. Our work is related to contemporary email classification
systems which could employ machine learning and natural
language processing. We propose a novel detection system for phishing emails
based on recurrent neural networks (RNNs) with attention mechanisms. Our evaluation indicates that our
RNN system outperforms many of the state-of-the-art tools.

In this paper, Section 2 presents alternative approaches to automated detection of phishing emails and literature on current machine learning approaches.
Section 3 details our methodology and feature selection for the RNN, while Section 4 describes the system implementation. Section 5 discusses the evaluation
of the system and Section 6 concludes the paper.

## 2 Current landscape

Specialised algorithms to classify email as phishing and spam (unsolicited email)
vs ham (not spam email) have been the focus of research since the beginning of
unsolicited email. Filtering phishing emails is a subcategory of the general
problem of spam filtering. As such, most email classifiers, hence filters, treat
relatively harmless spam equivalently to dangerous phishing emails.
Chandrasekaran et al., propose that phishing filtering needs to be treated
separately from the bulk spam filtering [10]. Phishing emails mimic ham emails,
in that they want to raise confidence in its perfectly legitimate origin. Their
proposed classifier uses 23 style marker features including the number of words
W, number of characters C and number of words per character W/C aliased
as vocabulary richness. The authors report results of up to perfect classification, with the accuracy dropping by 20% with the removal of the two structure
features. Although the experiment used only a small corpus of 400 emails, the
results demonstrate the importance of language, layout and structure of emails
in phishing classification.

PILFER is an ML-based phishing email classifier proposed by Fette et al. [9].
The authors identified a subset of only ten (from the hundreds of features popularly used to classify spam) that best distinguish phishing emails from ham.
PILFER outperformed the trained version of Apache’s SpamAssassin [12] at
classifying phishing emails, with the false negative rate reduced by a factor of ten. This result demonstrates that having specialised features for the task in
prominence to general email classification features improves phishing classification.

Bergholz et al. [6] build on top of this work by introducing advanced features
for phishing classification. The authors note the statistical insignificance of classification improvement through variation of the classifying algorithm itself. They
conclude that a statistically significant improvement is possible by invention of
better features. Bergholz et al. develop two sets of advanced features based on
unsupervised learning algorithms to complement 27 basic features commonly
used in spam detection. The advanced features are the Dynamic Markov Chain
(DMC) model features, and Latent Topic Model Features: word-clusters of topics
based on Latent Dirichlet Allocation (LDA). The best results occur when the
advanced features are used in conjunction with the basic features, achieving the
state-of-the-art [6].

Halgas et al. show us that traditional feature engineering doesn't perform as well as language modelling approaches to email classification. [3] This suggests that our intuitive understanding of what constitutes a phishing email may be
wrong. This could be due to the changing nature of phishing trends. By using natural language processing and deep learning techniques such as recurrent neural networks, they are able to capture the vocabulary richness of the email and subject. Their approach requires no human input and outperforms the state-of-the-art from manual expert feature construction.

## 3 Approach

The classification task is to identify an email as phishy or ham. If the email is phishy, we exclude it from delivery.
Due to the nature of phishing emails as well as implications on over-filtering, we
emphasize both precision and recall (later defined in Section 5) as the criteria for successful
classification. We believe in striking a balance between correctly classifying ham emails vs classifying phishing emails as phish. Too many false positives, or misclassifying ham email as phishing, and users will wonder where the emails went and question our phishing filtering. Too many false negatives and we run the risk of letting bad actors succeed in stealing credentials and other valuable information.

The machine learning approach to classification is to automatically establish
a function `f` that determines the desired class based on the input of a representation `x` of an email. The function f is parameterized by values `θ`. During the training phase, the parameter values θ are determined
to establish a relationship between the input `x` and class label `y` based on
a training set of pre-classified samples and a suitable optimisation criterion. In this sense, the ML approach is to extrapolate this relation between the observed sample points and class labels and use this relation on new unlabeled samples. To enforce the precision and recall balance requirement, we
will be optimising for the F-1 score.

### 3.1 Traditional ML vs Deep Learning

Whereas traditional machine learning has stages where we extract features and then train and classify input, deep learning approaches encapsulate the need for feature extraction through use of deep neural networks.

Neural Networks (NNs) are a computational model, in the quintessential example
of a multilayer perceptron (MLP) resembling a hierarchical network of units, or
neurones. The hierarchical structure intuitively gives NNs the capacity to extract
high-level features from simple data, i.e., to disentangle and winnow the factor
of variation in the NN input. This intuition of NN structure makes NNs suitable
for the task of representation learning, or automatic feature identification.
Recurrent Neural Networks (RNNs), the deepest of all learners, are a family of
NNs specialised for processing sequential data. Like Markov chain models, RNN
Catching the Phish 5
have the advantage of processing data in sequence, thus accounting for the order
of data. The input text is usually abstracted to a sequence of characters, words
or phrases. Undoubtedly, the order of words is valuable in language modelling.
RNNs form the backbone of current state-of-the-art language models, so an RNN
language model could form an accurate content-based classifier of emails.

### 3.2 Feature Identification

In order to identify features suitable for our models, we need to think about the representation of an email. Machine learning, after all, is all about intelligent representation.
A raw input representing an email as a long series of binary digits,
comprising the raw source code of an email in binary format, is unwieldy for
an algorithm to detect patterns. We hence use a more compact representation
by transforming the raw email into a feature vector x. Features should
characterize an email with respect to the current classification problem. The
relative inaccuracy of ML-based spam classifiers on the seemingly similar task of
phishing classification illustrates the need for specialised features for this task.
Features are most often identified by experts, in line with their intuitive understanding of “phishiness” or “hamness”. Halgas et al. [3] demonstrated that
such intuitively sound features often fail to inform the classification under discussion. On the other hand, structural features have empirically been indicative
of emails being ham or phish [9]. Based on this, we therefore follow the language
modelling approaches to the challenge of phishing classification which are viewed
as the most worthwhile [3].

Natural Language Processing (NLP) is the field of Computer Science studying human-machine interactions and, in particular, establishing and exploiting
language models. The rich structure and ambiguity of natural languages make
it difficult to identify and extract complex language features, such as the tone of
urgency in the email body.

One key idea behind text classification is document representation. Our initial approach includes a simple CountVectorizer and a TF-IDF Vectorizer. We process an email document and pass the counts/scores of the vocab into our classifier.

We can take this a step further and represent our document as a series of "word vectors" that embed semantic meaning (using some probabilistic technique). Creating word vectors is the process of taking a large corpus of text and creating a vector for each word such that words that share common contexts in the corpus are located in close proximity to one another in the vector space.

As explained earlier in the comparison between traditional ML and deep learning, the deep learning algorithm detects data patterns in the dataset without supervision or explicit expert advice. That is, the
training of the model determines, or learns, the features itself. Halgas et al. [3]
trained a recurrent neural network to detect ham or phishing
emails. We utilise similar NLP techniques in our system.

## 4 Design & Implementation

Our RNN classifier labels an input email as either a legitimate email or a phishing
attempt. In this section, we describe the procedure of transforming the raw email
source into a variable size vector of integers that is input to the RNN itself.

### 4.1 Email Text Preprocessing

Our binary classification RNN model takes sequences of integer values as input
and outputs a value between 0 and 1. We abstract the computer-native copy of
an email as a sequence of bytes into the high-level representation as a sequence of
symbol and word tokens, represented as unique integers. It is customary to ‘feed’
RNNs with an n-gram representation of the abstracted text. Due to the small
size of our dataset, our dictionary of n-grams would contain very few repetitive
phrases of n words for values n ≥ 2. For the balance of token expressiveness,

and vocabulary size, we choose to represent emails as sequences of 1-grams, or
single-word tokens.
Note that our classifier only considers the text of emails in making its classification decision. Thus, effective features, such as those based on linked web
address analysis, are completely orthogonal to our classifier and thus are largely
complementary. As an initial step in preprocessing of the classified email, we
extract its text in plaintext format.

### 4.2 Tokenizing the Text

We seek flexibility in tokenizing the text through fine tuning the parameters of
the tokenizer, such as rules of what word or character sequences to represent
as the same token. The naïve approach of splitting on whitespace characters
does not generalise well to email tokenizing. Incautious or malicious salting, e.g.,
inconsistent whitespace or the ubiquity of special characters, form words unique
to an email. Considering such tokens would inherently lead to overfitting, based
on the presence of unique traits.
Our approach to tokenizing is that of adjusted word-splitting. First, we lowercase all characters in the email and remove all characters the RFC 3986 standard
does not allow to be present in a URL, i.e., we only keep the unreserved a-z, 0-9,
- . _ ~ and reserved : / ? # [ ] @ ! $ & ’ ( ) * + , ; = characters and the percentage
sign %. Although this step is motivated by ease of later identifying URLs for
the <url> token determination, we get the benefit of restricting our character
base cardinality to 61. The 60th character, which RFC 3986 does not allow in
URL but we do not immediately replace with whitespace, is the quote character ", which is often used in emails. Note, the 61st character is the whitespace
character.
We introduce four special tokens summed up in Table 4.2, and, nine tokens
for the special characters left, replacing dots, quotes and seven other special
characters with their respective tokens. Finally, we split the clean text into words,
serving as their individual tokens, and prepend and append the start <s> and
end <e> tokens, respectively, to the tidy sequence of tokens.

The final representation of the email includes only lowercase alphanumeric
words and tokens. Using a list of allowed characters, we aggressively parse the
text, mitigating the threat of the text exhibiting unexpected behaviour.

### 4.3 Traditional Random Forest and Logistic Regression Classifiers

### 4.4 Recurrent Neural Network Classifier

Our model is a simple RNN, consisting of an encoding layer, two recurrent layers,
and a linear output layer with a Softplus activation. Challenges of training deep
networks, of which RNNs are the deepest, motivate most of the design decisions
presented in this section.
We implement our recurrent layers with the long short-term memory (LSTM)
architecture [8]. LSTM is a gated recurrent neural layer architecture that, through
its carefully designed self-loops, has the capacity to learn long range dependencies. Due to its carefully crafted architecture, LSTMs
are resistant to the vanishing gradient problem [17]. As is the standard, we use
the tanh nonlinear activation on the cells’ output. We describe the choice of the
size of the hidden layer to section below, but we will choose the hidden state to
be 200 variables large.
The output h2 of the last LSTM cell of the second layer is input further up
the model. So that our model outputs a single variable pyb ∈ (0, 1) as required.
Since we are modelling a Bernoulli probability, we use the simplest linear layer
h2 7→ w⊺h2 + b = z,
consisting of a weight vector w and bias scalar b. The final output is obtained
by mapping the linear layer output scalar through the logistic sigmoid function
pyb = σ(z) := 1
1 + exp(−x)
∈ (0, 1)
to obtain the estimated probability of an email being phish.

### 4.5 Input sequence embedding

### 4.6 Cutout Pruning

## 5 Evaluation

### 5.1 Training

### 5.2 Evaluation Metrics

## 6 Conclusion

## References

* [[1] 2019 Verizon Data Breach Investigations Report](https://www.nist.gov/system/files/documents/2019/10/16/1-2-dbir-widup.pdf)
* [[2] 2019 Symantec Internet Security Threat Report (ISRT)](https://docs.broadcom.com/doc/istr-24-2019-en)
* [[3] Halgaš, L., Agrafiotis, I., Nurse, J. R. Catching the Phish: Detecting Phishing Attacks using Recurrent Neural Networks (RNNs). arXiv preprint arXiv:1908.03640v1, 2019.](https://arxiv.org/pdf/1908.03640.pdf)
* [[4] Metsis, V., Androutsopoulos, I., &amp; Paliouras, G. (2006). Spam Filtering with Naive Bayes – Which Naive Bayes? Retrieved from http://www2.aueb.gr/users/ion/docs/ceas2006_paper.pdf](http://www2.aueb.gr/users/ion/docs/ceas2006_paper.pdf)
* [[5] Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)
* [[6] Bergholz, A., Chang, J.H., Paaß, G., Reichartz, F., Strobel, S.: Improved phishing detection using model-based features. In: Proceedings of the Fifth Conference on Email and Anti-Spam. CEAS ’08 (August 2008)](https://www.ceas.cc/2008/papers/ceas2008-paper-44.pdf)
* [[7] Cho K., Merrienboer B., Gulcehre C., Bougares F., Schwenk H., Bahdanau D., Bengio Y. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. arXiv preprint arXiv:1406.1078v3, 2014.](https://arxiv.org/pdf/1406.1078v3.pdf)
* [[8] Hochreiter S., Schmidhuber J. Long Short-Term Memory. In: Neural Computation. pp.1735-1780. WWW '06](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735#.WIxuWvErJnw)
* [[9] Fette, I., Sadeh, N., Tomasic, A.: Learning to detect phishing emails. In: Proceedings of the 16th international conference on World Wide Web. pp. 649–656. WWW ’07, ACM (May 2007)](http://www2007.thewebconf.org/papers/paper550.pdf)
* [[10] Chandrasekaran, M., Narayanan, K., Upadhyaya, S.: Phishing email detection based on structural properties. In: Proceedings of the 9th Annual NYS Cyber Security Conference. NYSCSC ’06 (June 2006)](https://www.albany.edu/iasymposium/proceedings/2006/chandrasekaran.pdf)
* [[11] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. GloVe: Global Vectors for Word Representation. 2014.](https://nlp.stanford.edu/projects/glove/)
* [[12] SpamAssassin corpus](https://spamassassin.apache.org/old/publiccorpus/)
* [[13] Galassi A., Lippi M., Torroni P. Attention in Natural Language Processing. In: IEEE Transactions on Neural Networks and Learning Systems. 2020.](https://arxiv.org/pdf/1902.02181.pdf)
* [[14] Yasin A., Abuhasan A. An Intelligent Classification Model for Phishing Email Detection. 2016.](https://arxiv.org/pdf/1608.02196.pdf)
* [[15] Lee Y., Saxe J., Harang R. CatBERT: Context Aware Tiny BERT for Detecting Social Engineering Emails. arXiv preprint arXiv:2010.03484v1, 2020.](https://arxiv.org/pdf/2010.03484.pdf)
* [[16] Nazario, J. dataset](https://monkey.org/~jose/phishing/)
* [[17] Greff K., Srivastava R.K., Koutník J., Steunebrink B.R., Schmidhuber J. LSTM: A Search Space Odyssey. arXiv:1503.04069, 2017.](https://arxiv.org/abs/1503.04069)
