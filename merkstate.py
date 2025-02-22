import random
import math
import numpy as np
from typing import NewType
from numpy.typing import NDArray
from globalvars import WIN_X, WIN_Y, GRID_X, GRID_Y

Activation = NewType('Activation', NDArray[np.float64])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class MerkState:

    # ------------------------------------------------ #

    def __init__(self) -> None:

        # general implicit
        self.x: int = random.randint(0, GRID_X)
        self.y: int = random.randint(0, GRID_Y)
        self.angle: int = random.randint(0, 3)
        self.moving: int = 1
        self.turn: int = 0
        self.health: float = 1.0
        self.speed: int = 2

        # sensors
        self.costmap_grad_forw: float = 0.0
        self.costmap_grad_lat: float = 0.0
        self.costmap_value: float = 0.0

    # ------------------------------------------------ #

    def action(self) -> None:
        self.speed = 2
        self.angle = round(self.angle + self.turn) % 4
        if self.moving > 0:
            if self.angle == 0:
                self.y = self.y - 1
            if self.angle == 1:
                self.x = self.x + 1
            if self.angle == 2:
                self.y = self.y + 1
            if self.angle == 3:
                self.x = self.x - 1
                
        # Continuous case
        #self.x = self.x + speed * \
            #_activation_av(self.av) * math.sin(self.angle)
        #self.y = self.y - speed * \
            #_activation_av(self.av) * math.cos(self.angle)

    # ------------------------------------------------ #

    def get_pos(self) -> np.ndarray:
        x = (int)(self.x % GRID_X)
        y = (int)(self.y % GRID_Y)
        return np.array([x, y, self.angle], dtype=int)

    def set_activations(self, action: Activation) -> None:
        self.moving = round(action[0])
        self.turn = round(action[1]) - round(action[2])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def example() -> None:

    random.seed(5)
    a = MerkState()
    a.moving = 1
    for i in range(10):
        a.action()
        a.turn = (random.random()-0.5)*2.0
        print(a.angle)
        print(a.x)
        print(a.y)
