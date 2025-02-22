from typing import List, Optional, Tuple

# Returns : rest of the genome, the sequence read
def readsegment(gen: List[int]) -> Tuple[List[int], List[int]]:
    found = 0
    if len(gen) == 0:
        return ([], [])
    for i in range(0, len(gen), 2):
        if gen[i] != 0:
            found = 1
            break
    # i is the first 1 encountered OR at the end of the array gen
    if found == 0:
        i = i+2
    return (gen[i+1:], gen[1:i:2])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def readrule(gen: List[int]) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[List[int]], Optional[List[int]], List[int]]:
    genstore = gen
    if gen[0:2] != [1, 1]:
        return (None, None, None, None, gen[1:])
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
    return (predecessor, successor, extra, codes, rest)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class Rule:
    def __init__(self) -> None:
        self.predecessor: Optional[List[int]] = None
        self.successor: Optional[List[int]] = None
        self.extra: Optional[List[int]] = None
        self.codes: Optional[List[int]] = None

    # ------------------------------------------------ #

    def build_from_adn(self, adn: List[int]) -> List[int]:
        predecessor, successor, extra, codes, rest = readrule(adn)
        if predecessor is not None:
            self.predecessor = predecessor
            self.successor = successor
            self.extra = extra
            self.codes = codes
        return rest

    # ------------------------------------------------ #

    def isRule(self) -> bool:
        return self.predecessor is not None

    # ------------------------------------------------ #

    def print(self) -> None:
        print("Pred: ", self.predecessor)
        print("Succ: ", self.successor)
        print("Extra: ", self.extra)
        print("Codes: ", self.codes)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
