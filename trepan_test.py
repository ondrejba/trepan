import unittest
import copy as cp
import numpy as np
import trepan


class TestTrepan(unittest.TestCase):

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

    def test_step_end(self):

        np.random.seed(2018)

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

        self.assertEqual(len(tp.queue), 0)

    def test_step_continue(self):

        np.random.seed(2018)

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

        self.assertEqual(len(tp.queue), 2)

    def test_train_impossible(self):
        np.random.seed(2018)

        data = np.concatenate([np.zeros(20, dtype=np.float32), np.ones(20, dtype=np.float32)], axis=0)
        data = np.expand_dims(data, axis=1)

        labels = np.concatenate([
            np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
            np.ones(10, dtype=np.float32) + 1, np.ones(10, dtype=np.float32) + 2
        ], axis=0)

        oracle = trepan.Oracle(lambda x: x[:, 0], trepan.Oracle.DataType.DISCRETE, 0.05, 0.05)
        tp = trepan.Trepan(data, labels, oracle, 15, 50)

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

        self.assertEqual(internal_count, 15)

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


class TestOracle(unittest.TestCase):

    def test_gen_discrete(self):

        np.random.seed(2018)

        f1 = np.concatenate([np.zeros(75, dtype=np.float32), np.ones(25, dtype=np.float32)], axis=0)
        f2 = np.concatenate([np.zeros(1, dtype=np.float32), np.ones(99, dtype=np.float32)], axis=0)

        data = np.stack([f1, f2], axis=1)

        oracle = trepan.Oracle(lambda x: np.zeros_like(x), trepan.Oracle.DataType.DISCRETE, 0.05, 0.05)

        new_data = oracle.gen_discrete(data, [], 10000)

        f1_1 = np.sum((new_data[:, 0] == 0).astype(np.float32))
        f1_2 = np.sum((new_data[:, 0] == 1).astype(np.float32))

        f2_1 = np.sum((new_data[:, 1] == 0).astype(np.float32))
        f2_2 = np.sum((new_data[:, 1] == 1).astype(np.float32))

        p0_0 = f1_1 / (f1_1 + f1_2)
        p1_0 = f2_1 / (f2_1 + f2_2)

        self.assertTrue(0.74 <= p0_0 <= 0.76)
        self.assertTrue(0.009 <= p1_0 <= 0.011)
