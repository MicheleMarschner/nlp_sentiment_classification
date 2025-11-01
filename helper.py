import matplotlib.pyplot as plt

from model.naivebayes import NaiveBayes, features1, features2
from model.logreg import LogReg, featurize
from evaluation import accuracy, f_1
import numpy as np


def train_smooth(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) Re-train Naive Bayes while varying smoothing parameter k,
    #         then evaluate on test_data.
    #         2) Plot a graph of the accuracy and/or f-score given
    #         different values of k and save it, don't forget to include
    #         the graph for your submission.

    ######################### STUDENT SOLUTION #########################
    
    k_values = [0.01, 0.1, 0.5, 1, 2, 5, 10]
    accs = []
    f1_score = []
    for k in k_values:
        nb = NaiveBayes.train(train_data, k)
        accs.append(accuracy(nb, test_data))
        f1_score.append(f_1(nb, test_data))

    plt.figure()
    plt.plot(k_values, accs, marker='o', label='Accuracy')
    plt.plot(k_values, f1_score, marker='s', label='Macro-F1')
    plt.xlabel('Smoothing parameter k')
    plt.ylabel('Score')
    plt.title('Naive Bayes performance vs. smoothing (k)')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    out_path = 'nb_smoothing_k_vs_metrics.png'
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    ####################################################################



def train_feature_eng(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) Improve on the basic bag of words model by changing
    #         the feature list of your model. Implement at least two
    #         variants using feature1 and feature2
    ########################### STUDENT SOLUTION ########################
    variants = [
        ("Stopwords+FreqFilter", features1),
        ("Bigrams",     features2),
    ]
    k = 1  # pick your preferred smoothing (or make this a loop if you want)

    names, accs, f1s = [], [], []

    for name, feat_fn in variants:
        # Each featuresX returns a processed dataset: list[(tokens, label)]
        feature_data = feat_fn(train_data, k)
        nb = NaiveBayes.train(feature_data, k)

        # Evaluate
        acc = accuracy(nb, test_data)
        f1  = f_1(nb, test_data)

        names.append(name)
        accs.append(acc)
        f1s.append(f1)
        print(f"{name:24s}  acc={acc:.4f}  macro-F1={f1:.4f}")

    # ---- Plot & save ----
    x = np.arange(len(names))
    w = 0.35
    plt.figure()
    plt.bar(x - w/2, accs, width=w, label="Accuracy")
    plt.bar(x + w/2, f1s,  width=w, label="Macro-F1")
    plt.xticks(x, names, rotation=15, ha="right")
    plt.ylabel("Score")
    plt.title(f"Naive Bayes: Feature Engineering (k={k})")
    plt.legend()
    plt.tight_layout()
    out_path = "nb_feature_eng.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot to {out_path}")

    # Return for potential downstream checks
    return {"names": names, "accuracy": accs, "macro_f1": f1s, "plot_path": out_path}
    #####################################################################



def train_logreg(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) First, assign each word in the training set a unique integer index
    #         with `buildw2i()` function (in model/logreg.py, not here)
    #         2) Now that we have `buildw2i`, we want to convert the data into
    #         matrix where the element of the matrix is 1 if the corresponding
    #         word appears in a document, 0 otherwise with `featurize()` function.
    #         3) Train Logistic Regression model with the feature matrix for 10
    #         iterations with default learning rate eta and L2 regularization
    #         with parameter C=0.1.
    #         4) Evaluate the model on the test set.
    ########################### STUDENT SOLUTION ########################

    X_train, Y_train = featurize(train_data, train_data)
    X_test, Y_test = featurize(test_data, train_data)

    logreg = LogReg(eta=0.01, num_iter=10, lambda_reg=0.1)
    logreg.train(X_train, Y_train)

    test_data_oneHot = list(zip(X_test, Y_test))

    print("Accuracy: ", accuracy(logreg, test_data_oneHot))
    print("F_1: ", f_1(logreg, test_data_oneHot))
    #####################################################################
