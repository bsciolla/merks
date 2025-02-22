import unittest
import random
from genome import Genome, weight_node, sort_unique

class TestGenomeBasics(unittest.TestCase):
    def test_weight_node(self):
        self.assertEqual(weight_node([]), 0)
        self.assertEqual(weight_node([0]), 1)
        self.assertEqual(weight_node([1]), 2)
        self.assertEqual(weight_node([0, 0]), 3)
        self.assertEqual(weight_node([0, 1]), 4)
        self.assertEqual(weight_node([1, 0]), 5)
        self.assertEqual(weight_node([1, 1]), 6)
        self.assertEqual(weight_node([0, 0, 0]), 7)
        self.assertEqual(weight_node([1, 1, 1]), 14)
        self.assertEqual(weight_node([0, 0, 0, 0]), 15)
        self.assertEqual(weight_node([0, 0, 0, 0, 0]), 31)

    def test_sort_unique(self):
        weights = [3, 1, 4, 1, 5]
        result = sort_unique(weights)
        self.assertEqual(result, [1, 0, 2, 4])

class TestGenome(unittest.TestCase):
    def test_genome_creation(self):
        random.seed(4)
        genome = Genome()
        genome.make_rules()
        genome.clear_rules()
        
        self.assertEqual(genome.rules[0].predecessor, [0])
        self.assertEqual(genome.rules[0].successor, [])
        self.assertEqual(genome.rules[0].extra, [1])
        self.assertEqual(genome.rules[0].codes, [0, 1, 1, 0])

if __name__ == '__main__':
    unittest.main()
