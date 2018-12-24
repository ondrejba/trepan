import functools
import itertools
import copy as cp
import numpy as np
from enum import Enum


class Rule:

    class SplitType(Enum):

        ABOVE = 1
        BELOW = 2

    def __init__(self, feature_idx, split_value, split_type):

        self.num_required = 1
        self.splits = [(feature_idx, split_value, split_type)]
        self.blacklist = {feature_idx}
        self.score = None

    def add_split(self, feature_idx, split_value, split_type):

        if feature_idx not in self.blacklist:
            self.splits.append((feature_idx, split_value, split_type))
            self.blacklist.add(feature_idx)
            return True

        return False

    def get_masks(self, data):

        split_masks = []
        mask = np.zeros(data.shape[0], dtype=np.bool)

        for split in self.splits:

            if split[2] == self.SplitType.ABOVE:
                split_mask = data[:, split[0]] >= split[1]
            else:
                split_mask = data[:, split[0]] < split[1]

            split_masks.append(split_mask)

        for num_required in range(1, self.num_required + 1):

            combs = list(itertools.combinations(split_masks, num_required))

            for comb in combs:
                comb_mask = functools.reduce(np.logical_and, comb)
                mask[comb_mask] = True

        mask_a = mask
        mask_b = np.logical_not(mask)

        return mask_a, mask_b


class Beam:

    def __init__(self, width):

        self.width = width
        self.items = []
        self.scores = []

    def add_item(self, item, score):

        if len(self.items) < self.width:

            self.items.append(item)
            self.scores.append(score)

            return True

        else:

            min_score = min(self.scores)

            if score > min_score:

                idx = self.scores.index(min_score)

                self.scores[idx] = score
                self.items[idx] = item

                return True

            return False

    def get_lowest(self):

        min_score = min(self.scores)
        idx = self.scores.index(min_score)

        return self.items[idx]

    def get_highest(self):

        max_score = max(self.scores)
        idx = self.scores.index(max_score)

        return self.items[idx]


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
    best_rule = None

    for feature_idx in range(num_features):

        mask_a = data[:, feature_idx] >= 0.5
        mask_b = data[:, feature_idx] < 0.5

        labels_a = labels[mask_a]
        labels_b = labels[mask_b]

        ig = information_gain(labels, labels_a, labels_b, e_labels=e_labels)

        if best_ig is None or best_ig < ig:

            best_ig = ig
            best_rule = Rule(feature_idx, 0.5, Rule.SplitType.ABOVE)
            best_rule.score = ig

    return best_rule


def beam_search(first_rule, data, labels, beam_width=2, feature_blacklist=None):

    # don't use the same feature twice on the path to this node
    if feature_blacklist is not None:
        feature_blacklist.add(first_rule.splits[0][0])
    else:
        feature_blacklist = {first_rule.splits[0][0]}

    num_features = data.shape[1]
    beam = Beam(beam_width)
    beam.add_item(first_rule, first_rule.score)

    beam_changed = True
    while beam_changed:
        beam_changed = False

        for rule in beam.items:

            for feature_idx in range(num_features):

                for split_type in [Rule.SplitType.ABOVE, Rule.SplitType.BELOW]:

                    if feature_idx in feature_blacklist:
                        continue

                    for op_idx in range(2):

                        tmp_rule = cp.deepcopy(rule)

                        if not tmp_rule.add_split(feature_idx, 0.5, split_type):
                            # don't use the same feature twice in a rule
                            continue

                        if op_idx == 1:
                            tmp_rule.num_required += 1

                        mask_a, mask_b = tmp_rule.get_masks(data)
                        tmp_rule.score = information_gain(labels, labels[mask_a], labels[mask_b])

                        beam_changed = beam.add_item(tmp_rule, tmp_rule.score)

            if beam_changed:
                break

    best_rule = beam.get_highest()
    return best_rule
