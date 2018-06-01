

# Returns : rest of the genome, the sequence read
def readsegment(gen):
    found = 0
    if len(gen) == 0:
        return([], [])
    for i in range(0, len(gen), 2):
        if gen[i] != 0:
            found = 1
            break
    # i is the first 1 encountered OR at the end of the array gen
    if found == 0:
        i = i+2
    return(gen[i+1:], gen[1:i:2])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def readrule(gen):
    genstore = gen
    if gen[0:2] != [1, 1]:
        return(None, None, None, None, gen[1:])
    gen = gen[2:]
    gen, predecessor = readsegment(gen)
    gen, successor = readsegment(gen)
    gen, extra = readsegment(gen)
    if len(gen) < 4:
        predecessor = None
        successor = None
        extra = None
        codes = None
        rest = genstore[1:]
    else:
        codes = gen[:4]
        rest = gen[4:]
    return(predecessor, successor, extra, codes, rest)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Rule:

    def __init__(self):
        self.predecessor = None
        self.successor = None
        self.extra = None
        self.codes = None

    # ------------------------------------------------ #

    def build_from_adn(self, adn):
        predecessor, successor, extra, codes, rest = readrule(adn)
        if predecessor is not None:
            self.predecessor = predecessor
            self.successor = successor
            self.extra = extra
            self.codes = codes
        return(rest)

    # ------------------------------------------------ #

    def isRule(self):
        return(self.predecessor is not None)

    # ------------------------------------------------ #

    def print(self):
        print("Pred: ", self.predecessor)
        print("Succ: ", self.successor)
        print("Extra: ", self.extra)
        print("Codes: ", self.codes)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def example():
    b = Rule()
    rest = b.build_from_adn(
        [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1])
    assert(rest == [1, 1])
    assert(b.predecessor == [1, 0, 1])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_readsegment():

    a, b = readsegment([])
    assert(a == [])
    assert(b == [])
    a, b = readsegment([0, 1])
    assert(a == [])
    assert(b == [1])
    a, b = readsegment([0, 0])
    assert(a == [])
    assert(b == [0])
    a, b = readsegment([1, 0])
    assert(a == [0])
    assert(b == [])
    a, b = readsegment([1, 0, 1])
    assert(a == [0, 1])
    assert(b == [])
    a, b = readsegment([0, 0, 0, 1, 1])
    assert(a == [])
    assert(b == [0, 1])
    a, b = readsegment([0, 0, 0, 1, 1, 0])
    assert(a == [0])
    assert(b == [0, 1])
    a, b = readsegment([0, 1, 1, 0, 0])
    assert(a == [0, 0])
    assert(b == [1])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def test_readrule():
    pre, suc, extra, codes, rest = readrule([0, 1, 1])
    assert([pre, suc, extra, codes, rest] ==
           [None, None, None, None, [1, 1]])

    pre, suc, extra, codes, rest = readrule([1, 1, 1, 1, 1])
    assert([pre, suc, extra, codes, rest] ==
           [None, None, None, None, [1, 1, 1, 1]])

    pre, suc, extra, codes, rest = readrule([1, 1, 1, 1, 1, 1, 0, 0, 1])
    assert([pre, suc, extra, codes, rest] ==
           [[], [], [], [1, 0, 0, 1], []])

    pre, suc, extra, codes, rest = readrule(
        [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1])
    assert([pre, suc, extra, codes, rest] ==
           [[1], [0], [1], [1, 0, 1, 1], []])

    pre, suc, extra, codes, rest = readrule(
        [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1])
    assert([pre, suc, extra, codes, rest] ==
           [[1], [0], [1], [1, 0, 1, 1], [1, 1]])

    pre, suc, extra, codes, rest = readrule(
        [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1])
    assert([pre, suc, extra, codes, rest] ==
           [[1, 0, 1], [1], [1], [0, 0, 1, 1], [1, 1]])

    pre, suc, extra, codes, rest = readrule(
        [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1])
    assert([pre, suc, extra, codes, rest] ==
           [[1, 0, 1], [0, 1], [], [0, 0, 1, 1], [1, 1]])
