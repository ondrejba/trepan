import numpy as np


def information_gain(labels, labels_a, labels_b):

    p_a = len(labels_a) / len(labels)
    p_b = len(labels_b) / len(labels)

    e_labels = entropy(labels)
    e_labels_a = entropy(labels_a)
    e_labels_b = entropy(labels_b)

    return e_labels - (p_a * e_labels_a + p_b * e_labels_b)


def entropy(labels):

    classes = np.unique(labels)
    value = 0

    for cls in classes:
        p_cls = np.sum(labels == cls) / len(labels)
        value -= p_cls * np.log2(p_cls)

    return value
