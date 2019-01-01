import functools
import itertools
import time
import copy as cp
import numpy as np
from scipy.stats import chi2
from scipy.stats import norm
from enum import Enum


class Trepan:

    # TODO: implement sequence of trees
    # TODO: implement same class pruning

    class Node:

        def __init__(self):

            self.rule = None
            self.left_child = None
            self.right_child = None
            self.parent = None

            self.leaf = True
            self.majority_class = None
            self.reach = None
            self.fidelity = None
            self.model = None
            self.blacklist = set()

        def get_upstream_constraints(self):

            constraints = []

            prev = self
            parent = self.parent

            while parent is not None:

                if parent.left_child is prev:
                    constraints.append(("left", parent.rule))
                else:
                    constraints.append(("right", parent.rule))

                prev = parent
                parent = parent.parent

            return constraints

        def get_total_reach(self):

            reach = self.reach
            parent = self.parent

            while parent is not None:

                reach *= parent.reach
                parent = parent.parent

            assert reach is not None

            return reach

    def __init__(self, data, labels, oracle, max_internal_nodes, min_samples, profile=False):

        self.oracle = oracle
        self.max_internal_nodes = max_internal_nodes
        self.min_samples = min_samples
        self.profile = profile

        self.num_internal_nodes = 0

        self.root = self.Node()
        self.root.leaf = False
        self.root.reach = 1

        start = time.time()

        synth_data, synth_labels, model = self.oracle.fill_data(data, [], self.min_samples)
        self.root.model = model

        end = time.time() - start

        if self.profile:
            print("generate data: {:.2f} seconds".format(end))

        self.queue = BestFirstQueue()
        self.queue.add(1, (self.root, data, labels, synth_data, synth_labels, set()))

    def train(self):

        step_idx = 0

        while not self.queue.is_empty() and self.num_internal_nodes < self.max_internal_nodes:

            if self.profile:
                print()
                print("step {:d}".format(step_idx + 1))

            start = time.time()

            self.step()

            end = time.time() - start

            if self.profile:
                print("total: {:.2f} seconds".format(end))

            step_idx += 1

    def predict(self, data):

        return self.predict_step(data, self.root)

    def predict_step(self, data, node):

        # edge cases
        if node is None:
            return None

        if data is None or len(data) == 0:
            return None

        if node.leaf:
            # leaf prediction
            cls = node.majority_class

            if cls is None:
                cls = -1

            labels = np.zeros(data.shape[0], dtype=np.int32) + cls

            return labels

        else:
            # internal node aggregation
            mask_a, mask_b = node.rule.get_masks(data)

            labels_a = self.predict_step(data[mask_a], node.left_child)
            labels_b = self.predict_step(data[mask_b], node.right_child)

            labels = np.zeros(data.shape[0], dtype=np.int32)

            assert labels_a is not None or np.sum(mask_a) == 0
            assert labels_b is not None or np.sum(mask_b) == 0

            if labels_a is not None:
                labels[mask_a] = labels_a

            if labels_b is not None:
                labels[mask_b] = labels_b

            return labels

    def step(self):

        node, data, labels, synth_data, synth_labels, blacklist = self.queue.dequeue()

        all_data, all_labels = self.join_datasets(data, labels, synth_data, synth_labels)

        start = time.time()

        best_simple_rule = find_best_binary_split(all_data, all_labels, feature_blacklist=blacklist)

        if best_simple_rule is None:
            # no more features to split
            return
        else:
            # the node will become an internal node
            self.num_internal_nodes += 1

        best_m_of_n_rule = beam_search(best_simple_rule, all_data, all_labels, feature_blacklist=blacklist)
        best_m_of_n_rule_pruned = prune_rule(best_m_of_n_rule, all_data, all_labels)

        end = time.time() - start

        if self.profile:
            print("find split: {:.2f} seconds".format(end))

        node.leaf = False
        node.rule = best_m_of_n_rule_pruned

        node.left_child = self.Node()
        node.right_child = self.Node()

        node.left_child.parent = node
        node.right_child.parent = node

        mask_a, mask_b = node.rule.get_masks(data)

        synth_mask_a, synth_mask_b = [], []
        if synth_data is not None:
            synth_mask_a, synth_mask_b = node.rule.get_masks(synth_data)

        new_blacklist = blacklist.union(node.rule.blacklist)

        for tmp_mask, tmp_synth_mask, tmp_node in zip([mask_a, mask_b], [synth_mask_a, synth_mask_b],
                                                      [node.left_child, node.right_child]):

            if np.sum(tmp_mask) > 0:

                start = time.time()

                tmp_synth_data, tmp_synth_labels, tmp_model = self.oracle.fill_data(
                    data[tmp_mask], tmp_node.get_upstream_constraints(),
                    max(self.min_samples, self.oracle.get_stop_num_samples())
                )
                tmp_node.model = tmp_model

                end = time.time() - start

                if self.profile:
                    print("generate data: {:.2f} seconds".format(end))

                _, tmp_all_labels = self.join_datasets(
                    data[tmp_mask], labels[tmp_mask], tmp_synth_data, tmp_synth_labels
                )

                tmp_majority_class, tmp_fidelity = self.get_majority_class(tmp_all_labels)

                tmp_node.majority_class = tmp_majority_class
                tmp_node.fidelity = tmp_fidelity
                tmp_node.reach = (np.sum(tmp_mask) + np.sum(tmp_synth_mask)) / all_data.shape[0]

                if tmp_fidelity < 1:

                    score = tmp_node.get_total_reach() * (1 - tmp_node.fidelity)

                    self.queue.add(
                        score, (tmp_node, data[tmp_mask], labels[tmp_mask], tmp_synth_data, tmp_synth_labels, new_blacklist)
                    )

    def join_datasets(self, data, labels, synth_data, synth_labels):

        if synth_data is not None and synth_labels is not None:
            all_data = np.concatenate([data, synth_data], axis=0)
            all_labels = np.concatenate([labels, synth_labels], axis=0)
        else:
            all_data, all_labels = data, labels

        return all_data, all_labels

    def get_majority_class(self, labels):

        classes = np.unique(labels)

        majority_class = None
        majority_fraction = None

        for cls in classes:

            fraction = np.sum(labels == cls) / len(labels)

            if majority_fraction is None or fraction > majority_fraction:
                majority_class = cls
                majority_fraction = fraction

        assert majority_class is not None and majority_fraction is not None
        return majority_class, majority_fraction

    def prune(self):

        self.prune_step(self.root)

    def prune_step(self, node):

        if node is None:
            return None, None

        if node.leaf:
            return node.majority_class, node.fidelity
        else:
            cls1, fid1 = self.prune_step(node.left_child)
            cls2, fid2 = self.prune_step(node.right_child)

            if cls1 == cls2 and fid1 == fid2 == 1.0:
                node.majority_class = cls1
                node.fidelity = fid1
                node.leaf = True
                node.rule = None
                node.left_child = None
                node.right_child = None

                return cls1, fid1
            else:
                return node.majority_class, node.fidelity


class BestFirstQueue:

    def __init__(self):

        self.queue = {}
        self.size = 0

    def add(self, score, payload):

        if score not in self.queue:
            self.queue[score] = [payload]
        else:
            self.queue[score].append(payload)

        self.size += 1

    def dequeue(self):

        assert not self.is_empty()

        max_score = np.max(list(self.queue.keys()))

        to_return = self.queue[max_score][0]
        del self.queue[max_score][0]

        if len(self.queue[max_score]) == 0:
            del self.queue[max_score]

        self.size -= 1
        return to_return

    def is_empty(self):

        return self.size == 0


class Rule:

    BACKTRACKING_TOLERANCE = 0.001

    class SplitType(Enum):

        ABOVE = 1
        BELOW = 2

    def __init__(self, feature_idx, split_value, split_type):

        self.num_required = 1
        self.splits = [(feature_idx, split_value, split_type)]
        self.blacklist = {feature_idx}
        self.score = None
        self.f1 = None
        self.f2 = None

    def add_split(self, feature_idx, split_value, split_type):

        if feature_idx not in self.blacklist:
            self.splits.append((feature_idx, split_value, split_type))
            self.blacklist.add(feature_idx)
            return True
        elif len(self.splits) > 1:
            # backtracking
            old_split = None
            old_split_idx = None

            for idx, split in enumerate(self.splits):
                if split[0] == feature_idx:
                    old_split = split
                    old_split_idx = idx

            assert old_split is not None and old_split_idx is not None

            if old_split[2] != split_type and np.abs(old_split[1] - split_value) <= self.BACKTRACKING_TOLERANCE:
                del self.splits[old_split_idx]
                self.blacklist.remove(feature_idx)
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

    def flip_splits(self):

        for idx in range(len(self.splits)):

            feature_idx, split_value, split_type = self.splits[idx]

            if split_type == self.SplitType.BELOW:
                split_type= self.SplitType.ABOVE
            else:
                split_type = self.SplitType.BELOW

            self.splits[idx] = (feature_idx, split_value, split_type)


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

        if len(self.scores) == 0:
            return None

        min_score = min(self.scores)
        idx = self.scores.index(min_score)

        return self.items[idx]

    def get_highest(self):

        if len(self.scores) == 0:
            return None

        max_score = max(self.scores)
        idx = self.scores.index(max_score)

        return self.items[idx]


class Oracle:

    class DataType(Enum):

        DISCRETE = 1
        CONTINUOUS = 2

    def __init__(self, predict, data_type, epsilon, delta):

        self.predict = predict
        self.data_type = data_type
        self.epsilon = epsilon
        self.delta = delta

    def fill_data(self, data, constraints, num_samples):

        if self.data_type == self.DataType.DISCRETE:
            model = DiscreteModel()
            model.fit(data)
        else:
            raise NotImplementedError("Density estimates not implemented.")

        data_synth = None
        labels_synth = None

        if len(data) < num_samples:
            data_synth = self.sample(data, constraints, num_samples - len(data), model)
            labels_synth = self.predict(data_synth)

        return data_synth, labels_synth, model

    def get_stop_num_samples(self):

        z = norm.ppf(1 - self.epsilon)
        return int(((z ** 2) * (1 - self.epsilon)) / self.epsilon)

    def sample(self, data, constraints, to_generate, model):

        # generate samples
        samples = np.zeros((to_generate, data.shape[1]), dtype=np.float32)

        for sample_idx in range(to_generate):

            samples[sample_idx] = self.sample_with_constraints(model, constraints)

        return samples

    def sample_with_constraints(self, model, constraints):

        orig_model = model

        # don't modify the original model or constraints
        model = cp.deepcopy(model)
        constraints = cp.deepcopy(constraints)

        disj_c = []

        # separate hard rules and disjunctive rules
        for side, rule in constraints:

            if side == "right":
                rule.flip_splits()

            if len(rule.splits) == 1:
                # register hard rules in the model
                assert rule.num_required == 1
                model.zero_by_split(*rule.splits[0])

            else:
                disj_c.append((side, rule))

        # register a random instance of disjunctive rules in the model
        for side, rule in disj_c:

            if side == "left":
                num_required = rule.num_required
            else:
                num_required = len(rule.splits) - rule.num_required + 1

            for _ in range(num_required):

                probabilities = np.zeros(len(rule.splits), dtype=np.float32)

                for split_idx, split in enumerate(rule.splits):

                    probabilities[split_idx] = model.split_probability(split[0], split[1], split[2])

                probabilities /= probabilities.sum()
                assert not np.any(np.isnan(probabilities))

                indices = list(range(len(rule.splits)))
                split_idx = np.random.choice(indices, p=probabilities)
                split = rule.splits[split_idx]

                model.zero_by_split(*split)

        if model.check_nans():

            for constraint in constraints:
                print(constraint[0], constraint[1].splits, constraint[1].num_required)

            print()
            print(orig_model.distributions)
            print(orig_model.values)
            print()

            print(model.distributions)
            print(model.values)
            print()

            raise ValueError("NAN in the feature probabilities.")

        return model.sample()


class DiscreteModel:

    def __init__(self):

        self.distributions = []
        self.values = []
        self.num_features = None

    def fit(self, data):

        assert len(data.shape) == 2

        self.num_features = data.shape[1]

        for feature_idx in range(self.num_features):

            values = sorted(np.unique(data[:, feature_idx]))
            counts = np.zeros(len(values), dtype=np.float32)

            for value_idx, value in enumerate(values):

                count = np.sum(data[:, feature_idx] == value)
                counts[value_idx] = count

            probs = counts / counts.sum()

            self.distributions.append(probs)
            self.values.append(values)

    def set_zero(self, feature_idx, value):

        assert 0 <= feature_idx < self.num_features
        assert value in self.values[feature_idx]

        value_idx = self.values[feature_idx].index(value)
        self.distributions[feature_idx][value_idx] = 0
        self.distributions[feature_idx] /= self.distributions[feature_idx].sum()

    def zero_by_split(self, feature_idx, split_value, split_type):

        for value in self.values[feature_idx]:
            if split_type == Rule.SplitType.BELOW and value > split_value:
                self.set_zero(feature_idx, value)
            elif split_type == Rule.SplitType.ABOVE and value <= split_value:
                self.set_zero(feature_idx, value)

    def split_probability(self, feature_idx, split_value, split_type):

        probability = 0

        for idx, value in enumerate(self.values[feature_idx]):
            if split_type == Rule.SplitType.BELOW and value <= split_value:
                probability += self.distributions[feature_idx][idx]
            elif split_type == Rule.SplitType.ABOVE and value > split_value:
                probability += self.distributions[feature_idx][idx]

        return probability

    def sample(self):

        sample = np.empty(self.num_features, dtype=np.float32)

        for feature_idx in range(self.num_features):

            value = np.random.choice(self.values[feature_idx], size=1, p=self.distributions[feature_idx])
            sample[feature_idx] = value

        return sample

    def check_nans(self):

        has_nans = False

        for feature_idx in range(self.num_features):

            if np.any(np.isnan(self.distributions[feature_idx])):
                has_nans = True
                print("feature {:d}".format(feature_idx), self.distributions[feature_idx], self.values[feature_idx])

        return has_nans


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


def find_best_binary_split(data, labels, feature_blacklist=None):

    num_features = data.shape[1]
    e_labels = entropy(labels)

    best_ig = None
    best_rule = None

    for feature_idx in range(num_features):

        if feature_blacklist is not None and feature_idx in feature_blacklist:
            continue

        # TODO: should consider every discrete value instead
        mask_a = data[:, feature_idx] >= 0.5
        mask_b = data[:, feature_idx] < 0.5

        labels_a = labels[mask_a]
        labels_b = labels[mask_b]

        ig = information_gain(labels, labels_a, labels_b, e_labels=e_labels)

        if best_ig is None or best_ig < ig:

            best_ig = ig
            best_rule = Rule(feature_idx, 0.5, Rule.SplitType.ABOVE)
            best_rule.score = ig
            best_rule.f1 = np.sum(mask_a)
            best_rule.f2 = np.sum(mask_b)

    return best_rule


def beam_search(first_rule, data, labels, beam_width=2, feature_blacklist=None):

    num_features = data.shape[1]
    beam = Beam(beam_width)
    beam.add_item(first_rule, first_rule.score)

    beam_changed = True
    while beam_changed:
        beam_changed = False

        for rule in beam.items:

            for feature_idx in range(num_features):

                for split_type in [Rule.SplitType.ABOVE, Rule.SplitType.BELOW]:

                    if feature_blacklist is not None and feature_idx in feature_blacklist:
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
                        tmp_rule.f1 = np.sum(mask_a)
                        tmp_rule.f2 = np.sum(mask_b)

                        worst_rule = beam.get_lowest()

                        if worst_rule is not None:
                            test = chi_square_rule(tmp_rule.f1, tmp_rule.f2, worst_rule.f1, worst_rule.f2)
                        else:
                            test = True

                        if test:
                            beam_changed = beam.add_item(tmp_rule, tmp_rule.score)

            if beam_changed:
                break

    best_rule = beam.get_highest()
    return best_rule


def prune_rule(rule, data, labels):

    change = True

    while len(rule.splits) > 1 and change:

        change = False

        for split_idx in range(len(rule.splits)):

            for op in [0, 1]:

                tmp_rule = cp.deepcopy(rule)

                if op == 1:
                    if tmp_rule.num_required <= 1:
                        continue
                    else:
                        tmp_rule.num_required -= 1

                del tmp_rule.splits[split_idx]

                mask_a, mask_b = tmp_rule.get_masks(data)
                tmp_rule.score = information_gain(labels, labels[mask_a], labels[mask_b])

                if tmp_rule.score >= rule.score:

                    rule = tmp_rule
                    change = True

            if change:
                break

    return rule


def chi_square_rule(f1, f2, ef1, ef2):

    chi = (((f1 - ef1) ** 2) / (f1 + ef1)) + (((f2 - ef2) ** 2) / (f2 + ef2))
    threshold = chi2.isf(0.05, 1)

    return chi > threshold


def chi_square_model(model1, model2):

    pass
