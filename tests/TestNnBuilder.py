import unittest
import numpy as np
import random
from GrowNeuralNetwork import GrowNeuralNetwork
from neuralnetwork import Neuralnetwork
from genome import Genome
from globalvars import BUILDING_CYCLES, SENSORS, ACTIONS

class TestNnBuilder(unittest.TestCase):
    def setUp(self):
        random.seed(2)
        self.nn = Neuralnetwork()
        self.gen = Genome()
        self.gen.make_clean_rules()
        self.test_sensors = [[0], [1], [0, 0]]
        self.test_actions = [[0, 1], [1, 0], [1, 1]]

    def test_build_nn(self):
        GrowNeuralNetwork(self.nn, self.gen, verbose=False)
        
        expected_neurons = [[0], [1], [0, 1], [1, 1, 1], [1, 1, 1], 
                          [1, 1, 1], [1, 1, 1], [1], [0]]
        self.assertEqual(self.nn.neurons, expected_neurons)

        expected_links = np.array([
            [0., 1., 0., 1., 0., 1., 0., 0., 0.],
            [0., 0., 0., 1., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 1., 1., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 0., 1., 0., 1., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0.],
            [1., 1., 0., 1., 1., 1., 0., 0., 0.],
            [1., 1., 0., 1., 1., 1., 1., 0., 0.],
            [1., 1., 0., 1., 1., 1., 1., 1., 0.]
        ])
        self.assertTrue(np.array_equal(self.nn.links, expected_links))

    def test_build_nn_with_verbose(self):
        # Just testing that verbose mode doesn't break anything
        GrowNeuralNetwork(self.nn, self.gen, verbose=True)
        self.assertGreater(len(self.nn.neurons), 0)

if __name__ == '__main__':
    unittest.main() 