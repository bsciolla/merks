import unittest
import numpy as np
import random
from neuralnetwork import Neuralnetwork, satisfies, find_best_match, create_index_but
from neuralnetwork import permute_connectivity, get_idx_list_for_sensors

class TestNeuralNetworkBasics(unittest.TestCase):
    def test_satisfies(self):
        self.assertFalse(satisfies([], []))
        self.assertTrue(satisfies([0], [0]))
        self.assertTrue(satisfies([0, 1], [0]))
        self.assertFalse(satisfies([1], [0]))
        self.assertFalse(satisfies([1, 0], [0]))
        self.assertTrue(satisfies([0, 1, 0], [0, 1]))
        self.assertFalse(satisfies([1, 1, 0], [0, 1]))
        self.assertFalse(satisfies([1, 0], [1, 0, 1]))

    def test_find_best_match(self):
        rule_names = [[1],
                     [0, 1],
                     [1, 0],
                     [0, 1, 0],
                     [1, 0, 1],
                     [0, 0, 0, 0, 1],
                     [0, 0, 1, 0, 1],
                     [0, 1, 0, 0, 0],
                     [0, 1, 1, 1, 1, 1],
                     [0, 1, 0, 0, 1, 0, 0]]
        self.assertEqual(find_best_match([1], rule_names), 0)
        self.assertIsNone(find_best_match([0], rule_names))
        self.assertEqual(find_best_match([0, 1], rule_names), 1)
        self.assertEqual(find_best_match([0, 1, 1], rule_names), 1)
        self.assertEqual(find_best_match([0, 1, 0, 1], rule_names), 3)
        self.assertEqual(find_best_match([0, 1, 1, 1, 1], rule_names), 1)
        self.assertEqual(find_best_match([1, 0, 1, 1, 1], rule_names), 4)

    def test_create_index_but(self):
        self.assertTrue(np.array_equal(
            create_index_but(10, [4, 6]),
            np.array([4, 6, 0, 1, 2, 3, 5, 7, 8, 9])
        ))
        self.assertTrue(np.array_equal(
            create_index_but(3, []),
            np.array([0, 1, 2])
        ))

    def test_permute_connectivity(self):
        mat = np.array([[i+j for i in range(3)] for j in range(10, 40, 10)])
        perm = np.array([0, 2, 1])
        mat = permute_connectivity(mat, perm)
        self.assertTrue(np.array_equal(mat, np.array(
            [[10, 12, 11],
             [30, 32, 31],
             [20, 22, 21]])))

    def test_double_fancy_indexing(self):
        u = np.zeros([6])
        v = np.array([0, 3, 4])
        y = - np.arange(6)
        z = np.array([1, 3, 5])
        u[v] = y[z]
        self.assertTrue(np.array_equal(u,
                                     np.array([-1., 0., 0., -3., -5., 0.])))

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.nn = Neuralnetwork()

    def test_create_neuron_and_links(self):
        self.nn.create_neuron([0, 1, 1, 0])
        self.nn.create_link([0, 1, 1, 0], [0, 0, 1], -1)
        self.nn.create_neuron([0, 1, 1, 0])
        self.nn.create_link([0, 1, 1, 0], [0, 0, 0], -1)
        self.nn.create_link([1], [0, 0, 0], -1)
        
        expected = np.array([[0., 0., 0., -1., 0.],
                           [1., 0., 0., -1., -1.],
                           [0., 1., 0., 0., 0.],
                           [0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.]])
        self.assertTrue(np.array_equal(self.nn.links, expected))

    def test_create_neuron_from_model(self):
        codes = [1, 0, 1, 0]
        neuron_idx = 1
        new_neuron = [0]
        self.nn.create_neuron_from_model(neuron_idx, new_neuron, codes)
        
        expected = np.array([[0., 0., 0., 0.],
                           [1., 0., 0., -1.],
                           [0., 1., 0., 1.],
                           [1., -1., 0., 0.]])
        self.assertTrue(np.array_equal(self.nn.links, expected))

    def test_sort_sensors_actions(self):
        self.nn.create_neuron_from_model(1, [0, 1], [1, 1, 1, 1])
        sensors = [[0], [0, 1]]
        actions = [[0, 0, 0], [1, 0]]
        self.nn.sort_sensors_actions(sensors, actions)
        
        self.assertEqual(self.nn.sensors_idx.tolist(), [0])
        self.assertEqual(self.nn.sensors_found, [False, True])
        self.assertEqual(self.nn.actions_idx.tolist(), [1])
        self.assertEqual(self.nn.actions_found, [True, False])
        self.assertEqual(self.nn.neurons, [[0, 1], [0, 0, 0], [0, 0, 1], [0, 1]])
        
        expected_links = np.array([[0., 1., 0., 1.],
                                 [0., 0., 1., 1.],
                                 [0., 0., 0., 0.],
                                 [0., 1., 1., 0.]])
        self.assertTrue(np.array_equal(self.nn.links, expected_links))

    def test_sort_sensors_actions2(self):
        self.nn.neurons = [[1, 1, 1], [0, 0, 0], [0, 1]]
        self.nn.links = np.array([[0., 0., 0.],
                                 [1., 0, 0.],
                                 [0., 1., 0.]])
        sensors = [[0], [1], [0, 0]]
        actions = [[0, 1], [1, 0]]
        self.nn.sort_sensors_actions(sensors, actions)

        self.assertEqual(self.nn.sensors_found, [False, False, False])
        self.assertEqual(self.nn.actions_idx.tolist(), [0])
        self.assertEqual(self.nn.actions_found, [True, False])
        self.assertEqual(self.nn.neurons, [[0, 1], [1, 1, 1], [0, 0, 0]])

        expected_links = np.array([[0., 0., 1.],
                                 [0., 0., 0.],
                                 [0., 1., 0.]])
        self.assertTrue(np.array_equal(self.nn.links, expected_links))

    def test_get_idx_list_for_sensors(self):
        random.seed(2)
        self.nn.create_neuron_from_model(1, [0, 1], [1, 1, 1, 1])
        sensors = [[0], [0, 1]]
        actions = [[0, 0, 0], [1, 0]]
        
        self.assertEqual(
            get_idx_list_for_sensors(self.nn, sensors),
            ([2], [False, True])
        )
        self.assertEqual(
            get_idx_list_for_sensors(self.nn, actions),
            ([1], [True, False])
        )

    def test_update_activations(self):
        self.nn.create_neuron_from_model(1, [0, 1], [1, 1, 1, 0])
        sensors = [[0], [0, 0, 0]]
        actions = [[0, 1], [1, 0]]
        self.nn.sort_sensors_actions(sensors, actions)
        
        actionval = self.nn.update_activations(np.array([0, 1]))
        np.testing.assert_array_equal(
            actionval,
            np.array([0.5, 0.]),
            "Action values do not match expected output"
        )
        
        np.testing.assert_array_equal(
            self.nn.activations,
            np.array([0., 0.5, 0., 0.5]),
            "Neuron activations do not match expected values"
        )

if __name__ == '__main__':
    unittest.main()