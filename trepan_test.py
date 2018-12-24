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
