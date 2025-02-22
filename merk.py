from typing import Optional
import neuralnetwork
import genome
import random
import GrowNeuralNetwork
import MerkState

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Merk:

    # ------------------------------------------------ #

    def __init__(self) -> None:
        self.neuralNetwork: neuralnetwork.Neuralnetwork = neuralnetwork.Neuralnetwork()
        self.genome: genome.Genome = genome.Genome(auto_initialize=False)
        self.merkState: MerkState.MerkState = MerkState.MerkState()

    # ------------------------------------------------ #

    def build_random_merk(self) -> None:
        self.genome.random_adn()
        self.genome.make_clean_rules()
        GrowNeuralNetwork.GrowNeuralNetwork(self.neuralNetwork, self.genome, verbose=False)

    # ------------------------------------------------ #

    def action(self) -> None:
        self.merkState.action()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def example() -> None:
    random.seed(5)
    a = Merk()
    a.build_random_merk()
    print(a.neuralNetwork.links)
