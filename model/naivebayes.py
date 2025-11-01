from collections import Counter, defaultdict
import math
from nltk.corpus import stopwords
import re

class NaiveBayes(object):

    ######################### STUDENT SOLUTION #########################
    # YOUR CODE HERE
    def __init__(self):
        """Initialises a new classifier."""
        """ Attributes saved
        - classes               list of all classes in train set
        - vocab                 dict of all words 
        - class_log_prior       XX
        - log_cond_prob         XX
        - trained               boolean
        """
    ####################################################################

    def predict(self, x):
        """Predicts the class for a document.

        Args:
            x: A document, represented as a list of words.

        Returns:
            The predicted class, represented as a string.
        """
        ################## STUDENT SOLUTION ########################
        if not self.trained:
            raise RuntimeError("NaiveBayes.predict() called before training")
        
        # Multinomial counts (works also if docs were binarized during training)
        counts = Counter(x)

        best_class, best_posterior_val = None, -float("inf")
        for c in self.classes:
            posterior_val = self.class_log_prior[c]
            cond_prob = self.log_cond_prob[c]

            for w, f in counts.items():
                if w in self.vocab:  # ignore unknown words
                    posterior_val += f * cond_prob[w]

            if posterior_val > best_posterior_val:
                best_posterior_val, best_class = posterior_val, c
                
        return best_class
        ############################################################


    @classmethod
    def train(cls, data, k=1):
        """Train a new classifier on training data using maximum
        likelihood estimation and additive smoothing.

        Args:
            cls: The Python class representing the classifier.
            data: Training data.
            k: The smoothing constant.

        Returns:
            A trained classifier, an instance of `cls`.
        """
        ##################### STUDENT SOLUTION #####################
        # Classes of train set
        labels = [y for _, y in data]
        classes = sorted(set(labels))

        # Total number of docs
        n_docs = len(data)
        #### print("n_docs", n_docs)

        # Build vocabulary from training data
        vocab = set()
        for tokens, _ in data:
            vocab.update(tokens)

        # Group documents by class
        docs_by_class = defaultdict(list)
        for tokens, y in data:
            docs_by_class[y].append(tokens)

        n_docs_by_class = Counter(labels)
        #### print("n_docs_by_class: ", n_docs_by_class)

        # Compute priors for each class
        class_log_prior = {c: math.log(n_docs_by_class[c] / n_docs) for c in classes}

        # Compute likelihoods with add-k smoothing
        # Class-specific token counts
        log_cond_prob = {c: {} for c in classes}
        for c in classes:
            bag_of_class = Counter()
            for tokens in docs_by_class[c]:
                bag_of_class.update(tokens) 
            total_words_per_class = sum(bag_of_class.values())

            denom = total_words_per_class + k * len(vocab)
            for w in vocab:
                num = bag_of_class[w] + k
                log_cond_prob[c][w] = math.log(num / denom)

        
        # Attach learned parameters to the model class
        cls.classes = classes
        cls.vocab = vocab
        cls.class_log_prior = class_log_prior
        cls.log_cond_prob = log_cond_prob
        cls.trained = True
        
        return cls()
        ############################################################


# Load NLTK's English stopwords
STOPWORDS = set(stopwords.words('english'))

def _clean(tokens):
    """Lowercase + keep only alphabetic words (min len 3)."""
    out = []
    for t in tokens:
        t = re.sub(r"[^a-zA-Z]", "", t.lower())
        if len(t) >= 3:
            out.append(t)
    return out
    

def features1(data, k=1):
    """
    Your feature of choice for Naive Bayes classifier.

    Args:
        data: Training data.
        k: The smoothing constant.

    Returns:
        Parameters for Naive Bayes classifier, which can
        then be used to initialize `NaiveBayes()` class
    """

    """
    A) remove stop words and frequent words
    + reduces noise and vocabulary size; forces the classifier to focus on more informative words.
    """
    ###################### STUDENT SOLUTION ##########################
    # YOUR CODE HERE
   # 1. Compute global word frequencies
    freq = Counter()
    for tokens, _ in data:
        cleaned = _clean(tokens)
        freq.update(cleaned)

    # 2. Define a cutoff for "very frequent" words
    threshold = 100  # you can tune this, e.g. 100 or top 2% of words
    frequent_words = {w for w, c in freq.items() if c >= threshold}

    # 3. Filter stopwords and frequent words
    filtered_data = []
    for tokens, label in data:
        words = _clean(tokens)
        words = [w for w in words if w not in STOPWORDS and w not in frequent_words]
        filtered_data.append((words, label))

    # 4. Return this processed data — your NB.train() will then use it
    return filtered_data
    ##################################################################


def features2(data, k=1):
    """
    Your feature of choice for Naive Bayes classifier.

    Args:
        data: Training data.
        k: The smoothing constant.

    Returns:
        Parameters for Naive Bayes classifier, which can
        then be used to initialize `NaiveBayes()` class
    """
    ###################### STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    filtered_data = []

    for tokens, label in data:
        # Create bigrams
        bigrams = []
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i+1]}"
            bigrams.append(bigram)


        # 3. Combine unigrams + bigrams
        all_features = tokens + bigrams

        filtered_data.append((all_features, label))

    return filtered_data
    ##################################################################

