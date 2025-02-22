from typing import Optional
import neuralnetwork
import genome
import random
import nnbuilder
import stagevars

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Merk:

    # ------------------------------------------------ #

    def __init__(self) -> None:
        self.nn: neuralnetwork.Neuralnetwork = neuralnetwork.Neuralnetwork()
        self.gen: genome.Genome = genome.Genome(auto_initialize=False)
        self.svars: stagevars.Stagevars = stagevars.Stagevars()

    # ------------------------------------------------ #

    def build_random_merk(self) -> None:
        self.gen.random_adn()
        self.gen.make_clean_rules()
        nnbuilder.build_nn(self.nn, self.gen, verbose=False)

    # ------------------------------------------------ #

    def action(self) -> None:
        self.svars.action()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def example() -> None:
    random.seed(5)
    a = Merk()
    a.build_random_merk()
    print(a.nn.links)
