
import rule
import random

from globalvars import ADN_LENGTH
from globalvars import MAX_BINARY_SIZE_FOR_NEURONS
from globalvars import MAX_HASH

from weightnode import weightnode as weight_node

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def truncate(neuron):
    if len(neuron) > MAX_BINARY_SIZE_FOR_NEURONS:
        return(neuron[:MAX_BINARY_SIZE_FOR_NEURONS])
    return(neuron)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_weight_node():
    assert(weight_node([]) == 0)
    assert(weight_node([0]) == 1)
    assert(weight_node([1]) == 2)
    assert(weight_node([0, 0]) == 3)
    assert(weight_node([0, 1]) == 4)
    assert(weight_node([1, 0]) == 5)
    assert(weight_node([1, 1]) == 6)
    assert(weight_node([0, 0, 0]) == 7)
    assert(weight_node([1, 1, 1]) == 14)
    assert(weight_node([0, 0, 0, 0]) == 15)
    assert(weight_node([0, 0, 0, 0, 0]) == 31)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def argsort(seq):
    # http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    # by unutbu
    return sorted(range(len(seq)),  key=seq.__getitem__)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def prune_list_by_index(lis, ind):
    return([lis[i] for i in ind])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def sort_unique(weight):
    inew = argsort(weight)
    iprime = [inew[0]]
    for idx, i in enumerate(inew):
        if idx > 0:
            if weight[i] > prev:
                iprime.append(i)
        prev = weight[i]
    return(iprime)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Genome:

    def __init__(self, auto_initialize=True):
        self.adn_length = ADN_LENGTH
        if auto_initialize is True:
            self.random_adn()
        else:
            self.adn = []
        self.rules = []

    # ------------------------------------------------ #

    def make_rules(self):

        local_adn = self.adn
        while (len(local_adn) > 0):

            nextrule = rule.Rule()
            local_adn = nextrule.build_from_adn(local_adn)

            if nextrule.isRule():
                self.rules.append(nextrule)

    # ------------------------------------------------ #

    def random_adn(self):
        self.adn = [random.randint(0, 1)
                    for i in range(self.adn_length)]

    # ------------------------------------------------ #

    def make_clean_rules(self):
        self.make_rules()
        self.clear_rules()

    # ------------------------------------------------ #

    def get_predecessors(self):
        return([rule.predecessor for rule in self.rules])

    # ------------------------------------------------ #

    def show_rules(self):
        for rule in self.rules:
            print("------------")
            rule.print()

    # ------------------------------------------------ #

    def clear_rules(self):
        newrules = []
        for rule in self.rules:
            if len(rule.predecessor) > 0:
                if len(rule.successor) + len(rule.extra) > 0:
                    newrules.append(rule)
        self.rules = newrules

        for rule in self.rules:
            rule.predecessor = truncate(rule.predecessor)
            rule.successor = truncate(rule.successor)
            rule.extra = truncate(rule.extra)

        # Sort by predecessor weight and keep only
        #  one rule for each predecessor
        pred_weight = [weight_node(self.rules[i].predecessor)
                       for i in range(len(self.rules))]

        iuniq = sort_unique(pred_weight)
        self.rules = prune_list_by_index(self.rules, iuniq)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def example_genome():
    b = Genome()
    b.make_rules()

    b.show_rules()
    b.clear_rules()
    print('\n')
    b.show_rules()


def test_genome():
    random.seed(4)
    b = Genome()
    b.make_rules()
    b.clear_rules()
    assert(b.rules[0].predecessor == [0])
    assert(b.rules[0].successor == [])
    assert(b.rules[0].extra == [1])
    assert(b.rules[0].codes == [0, 1, 1, 0])
