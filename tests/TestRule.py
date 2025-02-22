import unittest
from rule import Rule, readsegment, readrule

class TestRuleBasics(unittest.TestCase):
    def test_readsegment(self):
        a, b = readsegment([])
        self.assertEqual(a, [])
        self.assertEqual(b, [])

        a, b = readsegment([0, 1])
        self.assertEqual(a, [])
        self.assertEqual(b, [1])

        a, b = readsegment([0, 0])
        self.assertEqual(a, [])
        self.assertEqual(b, [0])

        a, b = readsegment([1, 0])
        self.assertEqual(a, [0])
        self.assertEqual(b, [])

        a, b = readsegment([1, 0, 1])
        self.assertEqual(a, [0, 1])
        self.assertEqual(b, [])

        a, b = readsegment([0, 0, 0, 1, 1])
        self.assertEqual(a, [])
        self.assertEqual(b, [0, 1])

        a, b = readsegment([0, 0, 0, 1, 1, 0])
        self.assertEqual(a, [0])
        self.assertEqual(b, [0, 1])

        a, b = readsegment([0, 1, 1, 0, 0])
        self.assertEqual(a, [0, 0])
        self.assertEqual(b, [1])

    def test_readrule(self):
        pre, suc, extra, codes, rest = readrule([0, 1, 1])
        self.assertEqual([pre, suc, extra, codes, rest],
                        [None, None, None, None, [1, 1]])

        pre, suc, extra, codes, rest = readrule([1, 1, 1, 1, 1])
        self.assertEqual([pre, suc, extra, codes, rest],
                        [None, None, None, None, [1, 1, 1, 1]])

        pre, suc, extra, codes, rest = readrule([1, 1, 1, 1, 1, 1, 0, 0, 1])
        self.assertEqual([pre, suc, extra, codes, rest],
                        [[], [], [], [1, 0, 0, 1], []])

        pre, suc, extra, codes, rest = readrule(
            [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1])
        self.assertEqual([pre, suc, extra, codes, rest],
                        [[1], [0], [1], [1, 0, 1, 1], []])

        pre, suc, extra, codes, rest = readrule(
            [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1])
        self.assertEqual([pre, suc, extra, codes, rest],
                        [[1], [0], [1], [1, 0, 1, 1], [1, 1]])

        pre, suc, extra, codes, rest = readrule(
            [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1])
        self.assertEqual([pre, suc, extra, codes, rest],
                        [[1, 0, 1], [1], [1], [0, 0, 1, 1], [1, 1]])

        pre, suc, extra, codes, rest = readrule(
            [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1])
        self.assertEqual([pre, suc, extra, codes, rest],
                        [[1, 0, 1], [0, 1], [], [0, 0, 1, 1], [1, 1]])

class TestRule(unittest.TestCase):
    def setUp(self):
        self.rule = Rule()

    def test_build_from_adn(self):
        rest = self.rule.build_from_adn(
            [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1])
        self.assertEqual(rest, [1, 1])
        self.assertEqual(self.rule.predecessor, [1, 0, 1])
        self.assertEqual(self.rule.successor, [0, 1])
        self.assertEqual(self.rule.extra, [])
        self.assertEqual(self.rule.codes, [0, 0, 1, 1])

    def test_isRule(self):
        self.assertFalse(self.rule.isRule())
        self.rule.predecessor = [1]
        self.assertTrue(self.rule.isRule())

if __name__ == '__main__':
    unittest.main() 