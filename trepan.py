import numpy as np


def information_gain(labels, labels_a, labels_b, e_labels=None):

    p_a = len(labels_a) / len(labels)
    p_b = len(labels_b) / len(labels)

    if e_labels is None:
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


def find_best_binary_split(data, labels):

    num_features = data.shape[1]
    e_labels = entropy(labels)

    best_ig = None
    best_mask_a = None
    best_mask_b = None

    for feature_idx in range(num_features):

        mask_a = data[:, feature_idx] >= 0.5
        mask_b = data[:, feature_idx] < 0.5

        labels_a = labels[mask_a]
        labels_b = labels[mask_b]

        ig = information_gain(labels, labels_a, labels_b, e_labels=e_labels)

        if best_ig is None or best_ig < ig:

            best_ig = ig
            best_mask_a = mask_a
            best_mask_b = mask_b

    return best_ig, best_mask_a, best_mask_b
