import numpy as np

class LogReg:
    def __init__(self, eta=0.01, num_iter=30, lambda_reg=0):
        self.eta = eta
        self.num_iter = num_iter
        self.lambda_reg = lambda_reg
        self.classes = None
        self.weight = None

    def softmax(self, inputs):
        """
        Calculate the softmax for the give inputs (array)
        :param inputs:
        :return:
        """

        exp_shifted = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    

    def train(self, X, Y):

        #################### STUDENT SOLUTION ###################

        n_samples, n_features = X.shape
        n_classes = len(np.unique(Y))
        batch_size = 100

        # Initialize weights
        self.weights = np.zeros((n_features, n_classes))
        self.classes = np.unique(Y)

        for epoch in range(self.num_iter):
            # Shuffle data at the start of each epoch
            idx = np.arange(n_samples)
            np.random.shuffle(idx)
            X_shuffled, Y_shuffled = X[idx], Y[idx]

            # Loop over mini-batches
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]

                # Compute predictions
                probs = self.p(X_batch)  # (batch_size, n_classes)

                # One-hot encode labels for this batch
                Y_onehot = np.zeros_like(probs)
                Y_onehot[np.arange(len(Y_batch)), Y_batch.astype(int)] = 1.0

                # Gradient of negative log-likelihood (averaged)
                grad_likelihood = np.dot(X_batch.T, (probs - Y_onehot)) / len(Y_batch)
                grad_reg = self.lambda_reg * self.weights
                gradient = grad_likelihood + grad_reg

                # Update weights (gradient descent)
                self.weights -= self.eta * gradient

        return None
        #########################################################


    def p(self, X):
        """
        Predict probabilities for each class.
        """
        # YOUR CODE HERE
        #     TODO:
        #         1) Fill in (log) probability prediction
        ################## STUDENT SOLUTION ########################
        X_reshaped = X if X.ndim == 2 else X[np.newaxis, :]
        logits = np.dot(X_reshaped, self.weights)
        probs = self.softmax(logits)   

        return probs if X.ndim == 2 else probs[0]
        ############################################################


    def predict(self, X):
        # YOUR CODE HERE
        #     TODO:
        #         1) Replace next line with prediction of best class
        ####################### STUDENT SOLUTION ####################
        probs = self.p(X)
        if probs.ndim == 1:                      # single example
            return int(np.argmax(probs))
        return np.argmax(probs, axis=1)
        #############################################################


def buildw2i(vocab):
    """
    Create indexes for 'featurize()' function.

    Args:
        vocab: vocabulary constructed from the training set.

    Returns:
        Dictionaries with word as the key and index as its value.
    """
    # YOUR CODE HERE
    #################### STUDENT SOLUTION ######################
    return {word: idx for idx, word in enumerate(vocab)}
    ############################################################


def featurize(data, train_data=None):
    """
    Convert data into X and Y where X is the input and
    Y is the label.

    Args:
        data: Training or test data.
        train_data: Reference data to build vocabulary from.

    Returns:
        Matrix X and Y.
    """
    ##################### STUDENT SOLUTION #######################
    
    vocab = set()
    for tokens, _ in train_data:
        vocab.update(tokens)

    w2i = buildw2i(vocab)

    label_set = sorted({y for _, y in train_data})
    label2i = {label: idx for idx, label in enumerate(label_set)}

    n = len(data)
    d = len(w2i)
    X = np.zeros((n, d), dtype=np.float32)
    Y = np.zeros(n, dtype=np.int64)

    for i, (tokens, label) in enumerate(data):
        for t in tokens:
            j = w2i.get(t)
            if j is not None:
                X[i, j] += 1.0
        Y[i] = label2i[label]

    return X, Y
    ##############################################################

