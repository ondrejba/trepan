import unittest
import random
import copy as cp
import numpy as np
import trepan


class TestTrepan(unittest.TestCase):

    def setUp(self):
        np.random.seed(2018)
        random.seed(2018)

    def test_node_get_upstream_constraints(self):

        p = trepan.Trepan.Node()
        p_lc = trepan.Trepan.Node()
        p_rc = trepan.Trepan.Node()
        p_lc_rc = trepan.Trepan.Node()

        p.left_child = p_lc
        p_lc.parent = p

        p.right_child = p_rc
        p_rc.parent = p

        p_lc.right_child = p_lc_rc
        p_lc_rc.parent = p_lc

        p.rule = 0
        p_lc.rule = 1
        p_lc_rc.rule = 2
        p_rc.rule = 3

        constraints = p_lc_rc.get_upstream_constraints()

        ref_constraints = [("right", 1), ("left", 0)]

        self.assertEqual(constraints, ref_constraints)

    def test_node_get_total_reach(self):

        p = trepan.Trepan.Node()
        p_lc = trepan.Trepan.Node()
        p_rc = trepan.Trepan.Node()
        p_lc_rc = trepan.Trepan.Node()

        p.left_child = p_lc
        p_lc.parent = p

        p.right_child = p_rc
        p_rc.parent = p

        p_lc.right_child = p_lc_rc
        p_lc_rc.parent = p_lc

        p.reach = 1
        p_lc.reach = 0.5
        p_lc_rc.reach = 0.7
        p_rc.rule = 0.3

        total_reach = p_lc_rc.get_total_reach()

        self.assertEqual(total_reach, p.reach * p_lc.reach * p_lc_rc.reach)

    def test_step_end(self):

        data = np.concatenate([np.zeros(100, dtype=np.float32), np.ones(100, dtype=np.float32)], axis=0)
        labels = cp.deepcopy(data)
        data = np.expand_dims(data, axis=1)

        oracle = trepan.Oracle(lambda x: x[:, 0], trepan.Oracle.DataType.DISCRETE, 0.05, 0.05)
        tp = trepan.Trepan(data, labels, oracle, 15, 50)

        tp.step()

        self.assertIsNotNone(tp.root.left_child)
        self.assertIsNotNone(tp.root.left_child.parent)
        self.assertIsNotNone(tp.root.right_child)
        self.assertIsNotNone(tp.root.right_child.parent)

        self.assertIsNone(tp.root.left_child.rule)
        self.assertIsNone(tp.root.right_child.rule)

        self.assertTrue(tp.queue.is_empty())

    def test_step_continue(self):

        data = np.concatenate([np.zeros(20, dtype=np.float32), np.ones(20, dtype=np.float32)], axis=0)
        data = np.expand_dims(data, axis=1)

        labels = np.concatenate([
            np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
            np.ones(10, dtype=np.float32) + 1, np.ones(10, dtype=np.float32) + 2
        ], axis=0)

        oracle = trepan.Oracle(lambda x: x[:, 0], trepan.Oracle.DataType.DISCRETE, 0.05, 0.05)
        tp = trepan.Trepan(data, labels, oracle, 15, 50)

        tp.step()

        self.assertIsNotNone(tp.root.left_child)
        self.assertIsNotNone(tp.root.left_child.parent)
        self.assertIsNotNone(tp.root.right_child)
        self.assertIsNotNone(tp.root.right_child.parent)

        self.assertIsNone(tp.root.left_child.rule)
        self.assertIsNone(tp.root.right_child.rule)

        self.assertEqual(tp.queue.size, 2)

    def test_train_impossible(self):

        data = np.random.uniform(0, 1, size=[100, 40])
        labels = np.random.randint(0, 30, size=100)

        oracle = trepan.Oracle(lambda x: x[:, 0], trepan.Oracle.DataType.DISCRETE, 0.05, 0.05)
        tp = trepan.Trepan(data, labels, oracle, 5, 50)

        tp.train()

        def count_nodes(node):

            internal = 0
            leafs = 0

            if node is not None:

                if node.leaf:
                    leafs += 1
                else:
                    internal += 1

                x, y = count_nodes(node.left_child)
                internal += x
                leafs += y

                x, y = count_nodes(node.right_child)
                internal += x
                leafs += y

            return internal, leafs

        internal_count, _ = count_nodes(tp.root)

        self.assertEqual(internal_count, 5)

    def test_entropy(self):

        labels_1 = np.ones(100, dtype=np.int32)
        e_1 = trepan.entropy(labels_1)
        self.assertEqual(e_1, 0.0)

        labels_1 = np.zeros(50)
        labels_2 = np.zeros(50) + 2
        e_2 = trepan.entropy(np.concatenate([labels_1, labels_2], axis=0))
        self.assertEqual(e_2, 1.0)

        labels_1 = np.zeros(75)
        labels_2 = np.zeros(25) + 2
        e_3 = trepan.entropy(np.concatenate([labels_1, labels_2], axis=0))
        self.assertTrue(0.8 <= e_3 <= 0.82)

    def test_information_gain(self):

        labels_a = np.zeros(75)
        labels_b = np.zeros(25) + 2

        labels = np.concatenate([labels_a, labels_b], axis=0)

        ig = trepan.information_gain(labels, labels_a, labels_b)

        self.assertTrue(0.8 <= ig <= 0.82)

    def test_beam_search(self):

        data_1 = np.stack([np.zeros(25), np.zeros(25)], axis=1)
        data_2 = np.stack([np.ones(25), np.ones(25)], axis=1)
        data_3 = np.stack([np.zeros(25), np.ones(25)], axis=1)
        data_4 = np.stack([np.ones(25), np.zeros(25)], axis=1)

        data = np.concatenate([data_1, data_2, data_3, data_4], axis=0)
        labels = np.concatenate([np.zeros(25), np.ones(75)], axis=0)

        best_first = trepan.find_best_binary_split(data, labels)

        best_n = trepan.beam_search(best_first, data, labels)

        self.assertGreater(best_n.score, best_first.score)
        np.testing.assert_almost_equal(best_n.score - trepan.entropy(labels), 0.0)

    def test_prune_rule_no_change(self):

        data_1 = np.stack([np.zeros(25), np.zeros(25)], axis=1)
        data_2 = np.stack([np.ones(25), np.ones(25)], axis=1)
        data_3 = np.stack([np.zeros(25), np.ones(25)], axis=1)
        data_4 = np.stack([np.ones(25), np.zeros(25)], axis=1)

        data = np.concatenate([data_1, data_2, data_3, data_4], axis=0)
        labels = np.concatenate([np.zeros(25), np.ones(75)], axis=0)

        best_first = trepan.find_best_binary_split(data, labels)

        best_n = trepan.beam_search(best_first, data, labels)
        best_n_pruned = trepan.prune_rule(best_n, data, labels)

        self.assertEqual(best_n, best_n_pruned)

        self.assertGreater(best_n_pruned.score, best_first.score)
        np.testing.assert_almost_equal(best_n_pruned.score - trepan.entropy(labels), 0.0)

    def test_prune_rule_change(self):

        data_1 = np.stack([np.zeros(25), np.zeros(25)], axis=1)
        data_2 = np.stack([np.ones(25), np.ones(25)], axis=1)
        data_3 = np.stack([np.zeros(25), np.ones(25)], axis=1)
        data_4 = np.stack([np.ones(25), np.zeros(25)], axis=1)

        data = np.concatenate([data_1, data_2, data_3, data_4], axis=0)
        labels = np.concatenate([np.zeros(25), np.ones(75)], axis=0)

        best_first = trepan.find_best_binary_split(data, labels)

        best_n = trepan.beam_search(best_first, data, labels)

        best_n.splits.append((1, 0.5, trepan.Rule.SplitType.ABOVE))

        best_n_pruned = trepan.prune_rule(best_n, data, labels)

        self.assertNotEqual(best_n, best_n_pruned)

        self.assertGreater(best_n_pruned.score, best_first.score)
        np.testing.assert_almost_equal(best_n_pruned.score - trepan.entropy(labels), 0.0)


class TestOracle(unittest.TestCase):

    def setUp(self):
        np.random.seed(2018)
        random.seed(2018)

    def test_sample_with_hard_constraints(self):

        f1 = np.concatenate([np.zeros(75, dtype=np.float32), np.ones(25, dtype=np.float32)], axis=0)
        f2 = np.concatenate([np.zeros(50, dtype=np.float32), np.ones(50, dtype=np.float32)], axis=0)
        f3 = np.concatenate([np.zeros(20, dtype=np.float32), np.ones(80, dtype=np.float32)], axis=0)

        data = np.stack([f1, f2, f3], axis=1)

        model = trepan.DiscreteModel()
        model.fit(data)

        constraints = [
            ("left", trepan.Rule(0, 0.5, trepan.Rule.SplitType.BELOW)),
            ("right", trepan.Rule(2, 0.5, trepan.Rule.SplitType.BELOW))
        ]

        oracle = trepan.Oracle(lambda x: x[:, 0], trepan.Oracle.DataType.DISCRETE, 0.05, 0.05)

        num_samples = 1000
        samples = []

        for _ in range(num_samples):

            samples.append(oracle.sample_with_constraints(model, constraints))

        samples = np.stack(samples)

        self.assertTrue(np.all(samples[:, 0] == 0))
        self.assertTrue(np.all(samples[:, 2] == 1))

        p1_0 = np.sum(samples[:, 1] == 0) / samples.shape[0]
        p1_1 = np.sum(samples[:, 1] == 1) / samples.shape[0]

        self.assertTrue(0.4 <= p1_0 <= 0.6)
        self.assertTrue(0.4 <= p1_1 <= 0.6)

    def test_sample_with_disj_constaints(self):

        f1 = np.concatenate([np.zeros(75, dtype=np.float32), np.ones(25, dtype=np.float32)], axis=0)
        f2 = np.concatenate([np.zeros(50, dtype=np.float32), np.ones(50, dtype=np.float32)], axis=0)
        f3 = np.concatenate([np.zeros(20, dtype=np.float32), np.ones(80, dtype=np.float32)], axis=0)

        data = np.stack([f1, f2, f3], axis=1)

        model = trepan.DiscreteModel()
        model.fit(data)

        rule = trepan.Rule(0, 0.5, trepan.Rule.SplitType.BELOW)
        rule.add_split(1, 0.5, trepan.Rule.SplitType.ABOVE)
        rule.add_split(2, 0.5, trepan.Rule.SplitType.BELOW)
        rule.num_required = 2

        constraints = [
            ("left", rule)
        ]

        oracle = trepan.Oracle(lambda x: x[:, 0], trepan.Oracle.DataType.DISCRETE, 0.05, 0.05)

        oracle.sample_with_constraints(model, constraints)


class TestDiscreteModel(unittest.TestCase):

    def setUp(self):
        np.random.seed(2018)
        random.seed(2018)

    def test_fit(self):

        f1 = np.concatenate([np.zeros(75, dtype=np.float32), np.ones(25, dtype=np.float32)], axis=0)
        f2 = np.concatenate([np.zeros(1, dtype=np.float32), np.ones(99, dtype=np.float32)], axis=0)

        data = np.stack([f1, f2], axis=1)

        model = trepan.DiscreteModel()
        model.fit(data)

        np.testing.assert_array_almost_equal(np.array([0.75, 0.25], dtype=np.float32), model.distributions[0])
        np.testing.assert_array_almost_equal(np.array([0.01, 0.99], dtype=np.float32), model.distributions[1])

        self.assertEqual([[0.0, 1.0], [0.0, 1.0]], model.values)

    def test_sample(self):

        f1 = np.concatenate([np.zeros(75, dtype=np.float32), np.ones(25, dtype=np.float32)], axis=0)
        f2 = np.concatenate([np.zeros(1, dtype=np.float32), np.ones(99, dtype=np.float32)], axis=0)

        data = np.stack([f1, f2], axis=1)

        model = trepan.DiscreteModel()
        model.fit(data)

        np.random.seed(2018)

        num_samples = 1000
        samples = []

        for _ in range(num_samples):
            samples.append(model.sample())

        samples = np.stack(samples, axis=0)

        p0_0 = np.sum(samples[:, 0] == 0) / samples.shape[0]
        p0_1 = np.sum(samples[:, 0] == 1) / samples.shape[0]
        p1_0 = np.sum(samples[:, 1] == 0) / samples.shape[0]
        p1_1 = np.sum(samples[:, 1] == 1) / samples.shape[0]

        self.assertTrue(0.7 <= p0_0 <= 0.8)
        self.assertTrue(0.2 <= p0_1 <= 0.3)
        self.assertTrue(0.001 <= p1_0 <= 0.1)
        self.assertTrue(0.8 <= p1_1 <= 1.0)

    def test_set_zero(self):

        f1 = np.concatenate([np.zeros(75, dtype=np.float32), np.ones(25, dtype=np.float32)], axis=0)
        f2 = np.concatenate([np.zeros(50, dtype=np.float32), np.ones(50, dtype=np.float32)], axis=0)

        data = np.stack([f1, f2], axis=1)

        model = trepan.DiscreteModel()
        model.fit(data)

        model.set_zero(0, 1)
        model.set_zero(1, 0)

        for _ in range(100):
            np.testing.assert_array_almost_equal(model.sample(), [0., 1.])

    def test_zero_by_split(self):

        f1 = np.concatenate([np.zeros(75, dtype=np.float32), np.ones(25, dtype=np.float32)], axis=0)
        f2 = np.concatenate([np.zeros(50, dtype=np.float32), np.ones(50, dtype=np.float32)], axis=0)

        data = np.stack([f1, f2], axis=1)

        model = trepan.DiscreteModel()
        model.fit(data)

        split1 = (0, 0.5, trepan.Rule.SplitType.BELOW)
        split2 = (1, 0.5, trepan.Rule.SplitType.ABOVE)

        model.zero_by_split(*split1)
        model.zero_by_split(*split2)

        for _ in range(100):
            np.testing.assert_array_almost_equal(model.sample(), [0., 1.])

    def test_split_probability(self):

        f1 = np.concatenate([np.zeros(75, dtype=np.float32), np.ones(25, dtype=np.float32)], axis=0)
        f2 = np.concatenate([np.zeros(50, dtype=np.float32), np.ones(50, dtype=np.float32)], axis=0)

        data = np.stack([f1, f2], axis=1)

        model = trepan.DiscreteModel()
        model.fit(data)

        split1 = (0, 0.5, trepan.Rule.SplitType.BELOW)
        split2 = (1, 0.5, trepan.Rule.SplitType.ABOVE)

        self.assertEqual(model.split_probability(*split1), 0.75)
        self.assertEqual(model.split_probability(*split2), 0.5)


class TestRule(unittest.TestCase):

    def setUp(self):
        np.random.seed(2018)
        random.seed(2018)

    def test_backtracking(self):

        rule = trepan.Rule(0, 0.7, trepan.Rule.SplitType.BELOW)
        rule.add_split(1, 0.5, trepan.Rule.SplitType.ABOVE)
        rule.add_split(2, 0.5, trepan.Rule.SplitType.ABOVE)

        self.assertFalse(rule.add_split(0, 0.7, trepan.Rule.SplitType.BELOW))
        self.assertFalse(rule.add_split(0, 0.5, trepan.Rule.SplitType.BELOW))
        self.assertFalse(rule.add_split(1, 0.7, trepan.Rule.SplitType.ABOVE))

        self.assertTrue(rule.add_split(0, 0.7, trepan.Rule.SplitType.ABOVE))
        self.assertEqual(len(rule.splits), 2)
        self.assertEqual(len(rule.blacklist), 2)

        self.assertTrue(rule.add_split(1, 0.5 - trepan.Rule.BACKTRACKING_TOLERANCE / 2, trepan.Rule.SplitType.BELOW))
        self.assertEqual(len(rule.splits), 1)
        self.assertEqual(len(rule.blacklist), 1)


class TestBestFirstQueue(unittest.TestCase):

    def setUp(self):
        np.random.seed(2018)
        random.seed(2018)

    def test_add_remove(self):

        queue = trepan.BestFirstQueue()
        queue.add(0.5, "a")
        queue.add(1, "b")
        queue.add(0, "c")

        output = [queue.dequeue() for _ in range(3)]

        self.assertEqual(output, ["b", "a", "c"])
        self.assertTrue(queue.is_empty())
        self.assertEqual(len(queue.queue), 0)

    def test_add_remove_overlap(self):

        queue = trepan.BestFirstQueue()
        queue.add(1, "a")
        queue.add(1, "b")
        queue.add(2, "c")

        output = [queue.dequeue() for _ in range(3)]

        self.assertEqual(output, ["c", "a", "b"])
        self.assertTrue(queue.is_empty())
        self.assertEqual(len(queue.queue), 0)
