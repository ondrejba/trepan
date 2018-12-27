import unittest
import numpy as np
import trepan


class TestTrepan(unittest.TestCase):

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

        oracle = trepan.Oracle(lambda x: np.zeros_like(x), 400, trepan.Oracle.DataType.DISCRETE)

        new_data = oracle.gen_discrete(data, [], 10000)

        f1_1 = np.sum((new_data[:, 0] == 0).astype(np.float32))
        f1_2 = np.sum((new_data[:, 0] == 1).astype(np.float32))

        f2_1 = np.sum((new_data[:, 1] == 0).astype(np.float32))
        f2_2 = np.sum((new_data[:, 1] == 1).astype(np.float32))

        p0_0 = f1_1 / (f1_1 + f1_2)
        p1_0 = f2_1 / (f2_1 + f2_2)

        self.assertTrue(0.74 <= p0_0 <= 0.76)
        self.assertTrue(0.09 <= p1_0 <= 0.11)
