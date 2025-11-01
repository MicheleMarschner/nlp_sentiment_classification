from collections import defaultdict

def accuracy(classifier, data):
    """Computes the accuracy of a classifier on reference data.

    Args:
        classifier: A classifier.
        data: Reference data.

    Returns:
        The accuracy of the classifier on the test data, a float.
    """
    ##################### STUDENT SOLUTION #########################
    if not data:
        return 0.0

    n_correct_preds = 0
    
    for token, c in data:
        predicted_class = classifier.predict(token)
        if (predicted_class == c):
            n_correct_preds += 1
    
    accuracy = n_correct_preds / len(data)

    return accuracy


def f_1(classifier, data):
    """Computes the F_1-score of a classifier on reference data.

    Args:
        classifier: A classifier.
        data: Reference data.

    Returns:
        The F_1-score of the classifier on the test data, a float.
    """
    ## theoretical input: https://iamirmasoud.com/2022/06/19/understanding-micro-macro-and-weighted-averages-for-scikit-learn-metrics-in-multi-class-classification-with-example/
    ##################### STUDENT SOLUTION #########################
    if not data:
        return 0.0

    gold = []
    preds = []
    for token, label in data:
        gold.append(label)
        pred = classifier.predict(token)
        preds.append(pred)

    classes = classifier.classes

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for y_true, y_hat in zip(gold, preds):
        if y_hat is None:
            # treat as a miss for the true class
            fn[y_true] += 1
            continue
        if y_hat == y_true:
            tp[y_true] += 1
        else:
            fp[y_hat] += 1
            fn[y_true] += 1

    # Macro-average F1 over classes
    f1s = []
    for c in classes:
        precision = (tp[c] / (tp[c] + fp[c])) if (tp[c] + fp[c]) > 0 else 0.0
        recall = (tp[c] / (tp[c] + fn[c])) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2*((precision * recall)/(precision + recall)) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    return sum(f1s) / len(f1s)

    ################################################################
