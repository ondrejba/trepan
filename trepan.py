import functools
import itertools
import copy as cp
import numpy as np
from enum import Enum


class Trepan:

    class Node:

        def __init__(self):

            self.rule = None
            self.left_child = None
            self.right_child = None

            self.leaf = True
            self.blacklist = set()

    def __init__(self, data, labels):

        self.root = self.Node()
        self.queue = [(self.root, data, labels, set())]

    def step(self):

        node, data, labels, blacklist = self.queue[0]
        del self.queue[0]

        # TODO: run oracle on data

        best_simple_rule = find_best_binary_split(data, labels)
        best_m_of_n_rule = beam_search(best_simple_rule, data, labels, feature_blacklist=blacklist)

        node.left = False
        node.rule = best_m_of_n_rule

        node.left_child = self.Node()
        node.right_child = self.Node()

        # TODO: determine if children should become leafs

        mask_a, mask_b = node.rule.get_masks(data)
        new_blacklist = blacklist.union(node.rule.blacklist)

        self.queue.append((node.left_child, data[mask_a], labels[mask_a], new_blacklist))
        self.queue.append((node.right_child, data[mask_b], labels[mask_b], new_blacklist))


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

        # TODO: add backtracking

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

    def match(self, features):

        matches = 0

        for split in self.splits:

            if split[2] == self.SplitType.ABOVE:
                if features[split[0]] >= split[1]:
                    matches += 1
            else:
                if features[split[0]] < split[1]:
                    matches += 1

        if matches >= self.num_required:
            return True

        return False


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


class Oracle:

    class DataType(Enum):

        DISCRETE = 1
        CONTINUOUS = 2

    def __init__(self, predict, min_samples, data_type):

        self.predict = predict
        self.min_samples = min_samples
        self.data_type = data_type

    def fill_data(self, data, labels, constraints):

        if data.shape[0] < self.min_samples:

            if self.data_type == self.DataType.DISCRETE:
                data_synth = self.gen_discrete(data, constraints, self.min_samples - data.shape[0])
            else:
                raise NotImplementedError("Density estimates not implemented.")

            labels_synth = self.predict(data_synth)

            data = np.concatenate([data, data_synth], axis=0)
            labels = np.concatenate([labels, labels_synth], axis=0)

        return data, labels

    def gen_discrete(self, data, constraints, to_generate):

        # create a histogram for each feature
        num_features = data.shape[1]
        counts = [{} for _ in range(num_features)]
        sums = np.zeros(num_features, dtype=np.float32)

        for feature_idx in range(num_features):

            values = np.unique(data[:, feature_idx])

            for value in values:

                count = np.sum((data[:, feature_idx] == value).astype(np.float32))
                counts[feature_idx][value] = count
                sums[feature_idx] += count

        for count_dict, count_sum in zip(counts, sums):
            for key in count_dict.keys():
                count_dict[key] /= count_sum

        probs = [[] for _ in range(num_features)]
        values = [[] for _ in range(num_features)]

        for idx, count_dict in enumerate(counts):
            for key, value in count_dict.items():
                probs[idx].append(value)
                values[idx].append(key)

        # generate samples
        samples = np.zeros((to_generate, data.shape[1]), dtype=np.float32)

        for sample_idx in range(to_generate):

            while True:

                # generate features
                for feature_idx in range(num_features):

                    feature = np.random.choice(values[feature_idx], size=1, replace=False, p=probs[feature_idx])
                    samples[sample_idx, feature_idx] = feature

                all_matched = True

                # check for constraints
                for constraint in constraints:
                    if not constraint.match(samples[sample_idx]):
                        all_matched = False

                if all_matched:
                    break

        return samples


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
