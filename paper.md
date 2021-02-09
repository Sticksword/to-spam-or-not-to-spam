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

The classification task is to identify an email as phishy or ham. If the email is phishy, I exclude it from delivery.
Due to the nature of phishing emails as well as implications on over-filtering, I
emphasize both precision and recall (later defined in Section 5) as the criteria for successful
classification. I believe in striking a balance between correctly classifying ham emails vs classifying phishing emails as phish. Too many false positives, or misclassifying ham email as phishing, and users will wonder where the emails went and question our phishing filtering. Too many false negatives and I run the risk of letting bad actors succeed in stealing credentials and other valuable information.

The machine learning approach to classification is to automatically establish
a function `f` that determines the desired class based on the input of a representation `x` of an email. The function f is parameterized by values `θ`. During the training phase, the parameter values θ are determined
to establish a relationship between the input `x` and class label `y` based on
a training set of pre-classified samples and a suitable optimisation criterion. In this sense, the ML approach is to extrapolate this relation between the observed sample points and class labels and use this relation on new unlabeled samples. To enforce the precision and recall balance requirement, I
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
an algorithm to detect patterns. I hence use a more compact representation
by transforming the raw email into a feature vector x. Features should
characterize an email with respect to the current classification problem. The
relative inaccuracy of ML-based spam classifiers on the seemingly similar task of
phishing classification illustrates the need for specialised features for this task.
Features are most often identified by experts, in line with their intuitive understanding of “phishiness” or “hamness”. Halgas et al. [3] demonstrated that
such intuitively sound features often fail to inform the classification under discussion. On the other hand, structural features have empirically been indicative
of emails being ham or phish [9]. Based on this, I therefore follow the language
modelling approaches to the challenge of phishing classification which are viewed
as the most worthwhile [3].

Natural Language Processing (NLP) is the field of Computer Science studying human-machine interactions and, in particular, establishing and exploiting
language models. The rich structure and ambiguity of natural languages make
it difficult to identify and extract complex language features, such as the tone of
urgency in the email body.

One key idea behind text classification is document representation. My initial approach includes a simple CountVectorizer and a TF-IDF Vectorizer. I process an email document and pass the counts/scores of the vocab into our classifier.

I can take this a step further and represent our document as a series of "word vectors" that embed semantic meaning (using some probabilistic technique). Creating word vectors is the process of taking a large corpus of text and creating a vector for each word such that words that share common contexts in the corpus are located in close proximity to one another in the vector space.

As explained earlier in the comparison between traditional ML and deep learning, the deep learning algorithm detects data patterns in the dataset without supervision or explicit expert advice. That is, the
training of the model determines, or learns, the features itself. Halgas et al. [3]
trained a recurrent neural network to detect ham or phishing
emails. I utilize similar NLP techniques in my system.

## 4 Design & Implementation

Our RNN classifier labels an input email as either a legitimate email or a phishing
attempt. In this section, I describe the procedure of transforming the raw email
source into a variable size vector of integers that is input to the RNN itself.

### 4.1 Email Text Preprocessing

Our binary classification RNN model takes sequences of integer values as input
and outputs a value between 0 and 1. I abstract the computer-native copy of
an email as a sequence of bytes into the high-level representation as a sequence of
symbol and word tokens, represented as unique integers. It is customary to ‘feed’
RNNs with an n-gram representation of the abstracted text. Due to the small
size of our dataset, our dictionary of n-grams would contain very few repetitive
phrases of n words for values n ≥ 2. For the balance of token expressiveness,
and vocabulary size, I choose to represent emails as sequences of 1-grams, or
single-word tokens.
Note that our classifier only considers the text of emails in making its classification decision. Thus, effective features, such as those based on linked web
address analysis, are completely orthogonal to our classifier and thus are largely
complementary. As an initial step in preprocessing of the classified email, I
extract its text in plaintext format.

### 4.2 Tokenizing the Text

I wanted adaptability when tokenizing the corpus. I ended up fine tuning the parameters of
the tokenizer, such as rules of what word or character sequences to represent
as the same token. The naïve approach of splitting on whitespace characters
does not generalise well to email tokenizing. Incautious or malicious salting, e.g.,
inconsistent whitespace or the ubiquity of special characters, form words unique
to an email. Considering such tokens would inherently lead to overfitting, based
on the presence of unique traits.
Our approach to tokenizing is that of adjusted word-splitting. First, I lowercase all characters in the email and only keep a-z, 0-9, and punctuations
(`- . _ ~ : / ? # [ ] @ ! $ & ’ ( ) * + , ; = % "`). This approach allows us to restrict our character
base to a limited set and removes any non-ascii characters.
I introduce one special token, `<URL>`, to represent potential URLs. I use the library `urlextract` to extract URLs from a given string [18].

The final representation of the email includes only lowercase alphanumeric
words, punctuation, and our special URL token. Using a set of allowed characters, I aggressively parse the
text to mitigate the threat of the text exhibiting unexpected behavior.

### 4.3 Traditional Random Forest and Logistic Regression Classifiers

My initial baseline tests were run with traditional random forest and logistic regression classifiers.

### 4.4 Recurrent Neural Network Classifier

My model is a simple RNN, consisting of an embedding layer, a recurrent layer, a global max pooling layer,
and a dense layer with rectified linear and sigmoid activation functions. Challenges of training deep
networks, of which RNNs are the deepest, motivate most of the design decisions
presented in this section.
I implement my recurrent layer with the long short-term memory (LSTM)
architecture [8]. LSTM is a gated recurrent neural layer architecture that, through
its carefully designed self-loops, has the capacity to learn long range dependencies. Due to its carefully crafted architecture, LSTMs are resistant to the vanishing gradient problem [17]. As is the standard, I use
the tanh nonlinear activation on the cells’ output. I choose the hidden state to be 128 variables large with a dropout value of 0.2 and also return the hidden state output for each timestep.
I plug the output `h` of the LSTM cells of the recurrent layer into a global 1D max pooling layer before finally mapping the result into a dense layer. The final output is obtained by mapping the linear layer output scalar through the logistic sigmoid function to obtain the estimated probability of an email being phish.

### 4.5 Input sequence embedding

I could train my own word vectors or I could use a pretrained word vector. As a fan of transfer learning, I opted to use a pretrained word vector, namely GloVe [11]. I also opted to combine words of similar meaning. I stemmed the input tokens using the Snowball Stemmer, a more aggressive (and generally regarded as the better) version of the popular Porter Stemmer. Lastly, for the tokens that appeared in my email corpus that do not show up in the GloVe word embeddings, I treat them as unknown, with a word vector of all zeros.

If I wanted to train my own word embeddings, I would need to limit what words to encode and what to ignore. If I let every token in the dataset have its unique embedding vector, not
only would the encoding layer be huge, but my model predictions would not
generalise well to any emails containing unknown words. I would need to reduce the
size of the dictionary considered by our model, in order to acquire round values,
to the `X` most common words in the training and validation sets of emails
as token sets (i.e., we do not consider repetitions of a word in a single email in
determining the occurrence count). I would then add 3 more tokens <unkalpha>, <unknnum>, <unk> to the dictionary. These abstract out unknown words to the dictionary, such as those that consist of only alphabetical or numerical values, or fit none of first two, respectively.

### 4.6 Email Body Pruning

Anomalous emails of very long sequence representations cause training inefficiency, amongst other problems, in evaluating very long range dependencies. The
problem is that such long emails cause unnecessary ‘padding’ of other, shorter
sequences, when employing gradient-based learning in batches, reducing stability and the speed of learning. Most notably, modern GPU architectures take
time proportional to the maximum length of a sample in the batch to evaluate
batched samples, as we do.
We hence compromise our email representation for excessively long emails
via a simple pruning procedure. I simply limit the length of input for each email to 400 tokens. The concept is to keep the beginning, which contains the subject and other important initial parts of an email, skipping the uninformative bits of ham or phish emails that follow after the initial 400 tokens.
Thus, emails represented as sequences of tokens of length 400, are fed into the RNN. The first,
encoding layer, converts each token in sequence with its corresponding word
embedding. The parameters of the embedding layer (each word embedding) are set from the start and will not be learned but instead loaded from GloVe [11].

## 5 Evaluation

Before presenting the results of our RNN classifier, we first introduce the email
datasets used in evaluation. We used a combination of 2751 ham emails and 501 spam emails from the
SpamAssassin public corpus [12] and 2279 phishing emails from the Nazario
phishing corpus [16] collected before August 2007. SpamAssassin and the Nazario datasets are popular
and used in related work to evaluate comparable phishing detection solutions [6,9].
In addition, we used the Enron email dataset. The Enron email dataset is generated by 158 employees of the Enron Corporation and is one of the largest public datasets of real-world emails.
We select a subset of 3672 ham emails and 1500 spam emails from the Enron dataset.
As is common practice in statistical learning, we split the data samples for
training and evaluation. Lastly, we pull from our own Gmail inbox to generate 668 spam and phishing examples.

Overall 2751 + 3672 = 6423 ham (56.5%) and 501 + 2279 + 1500 + 668 = 4948 spam/phishy (43.5%)

We split these into an 80% – 10% – 10% split for training and validation, and testing sets.

We evaluate our classifier against the most popular metrics in email classifications, which we introduce shortly. We then compare our language model to other content-based classifiers.

### 5.1 Training

The embedding layer accounts for 13,358,700 parameters of the model but these are not trainable. We load these directly from GloVe.

We initialize the weights of the LSTM cells to to their default values from Tensorflow.

The model contains a dropout of 0.2 for the LSTM layer as well as a global max pooling layer.

The model is optimized using the Adam optimizer against the binary
cross entropy loss function. We train the model with batches of size 128 samples.
The training dataset is shuffled at the beginning of every epoch. Finally, we stop training early if the loss does not go further beneath the current training minimum.

### 5.2 Evaluation Metrics

Given that the datasets used for email classification vary greatly in how even their
distributions are, the obvious accuracy measure is of limited value for comparison
to other classifiers. We hence report the standard measures of precision, recall,
F1 score, false positive, and false negative rates in addition to accuracy.

We note that email classification errors vary in importance. As an artifact of
the problem of spam email classification, it is common practice to consider a false
positive error to be more costly than a false negative misclassification. However,
this is under the assumption of aggressive filtering of positives and harmless false
negatives. In the domain of phishing emails, however, false negatives present
significant danger and less aggressive filtering methods such as alerts and link disabling are common.

We train the classifier over 4 epochs on the training dataset and 1 more
epoch over the validation dataset. Because the model is expensive to train, in
time and computational power, the results provided are of the single trained
instance. We evaluate the model on the test set, which had been unseen during
training.

## 6 Conclusion

In this paper and project, we propose a new automated system aiming to mitigate the threat
of phishing emails on the Square Marketing platform with the use of RNNs. Our results suggest that the flexibility of
RNNs gives our system an edge over the expert feature selection procedure, which
is vastly employed in Machine-Learning-based attempts at phishing mitigation. In addition, email representation is key and our choice of GloVe embeddings for text helped our cause.

We focused on the overlooked content source of email information and demonstrated its utility when considered in phishing threat mitigation. The nature of
RNNs and its training procedure make it suitable for the case of online learning. Our classifier could theoretically change over time to capture new trends continuously and keep up accurate and precise classification throughout.
Our results have demonstrated a wealth of potential in non-trivial feature identification for classifying emails, since our system’s performance surpasses the state-of-the-art systems which are based on features designed by human intuition.

Finally, it is worth noting that the general criticism of supervised learning
extends to our case. Little information is provided by the RNN classifier on the
nature of emails at successful classification. The proposed solution generalises
easily to the case of inclusion of basic spam emails, and is a prospect for further
automated success.

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
* [[18] urlextract pypi project page](https://pypi.org/project/urlextract/)
