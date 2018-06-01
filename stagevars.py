
import random
import math

from globalvars import WIN_X
from globalvars import WIN_Y


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class Stagevars:

    # ------------------------------------------------ #

    def __init__(self):
        
        # general implicit
        self.x = random.random()*WIN_X
        self.y = random.random()*WIN_Y
        self.angle = random.random()*math.pi*2.0
        self.av = 1
        self.turn = 0
        self.health = 1

        # sensors
        self.costmap_grad_forw = 0
        self.costmap_grad_lat = 0
        self.costmap_value = 0

    # ------------------------------------------------ #

    def action(self):
        speed = 2
        self.angle = self.angle + self.turn * math.pi/8.0
        self.x = self.x + speed * _activation_av(self.av) * math.sin(self.angle)
        self.y = self.y - speed * _activation_av(self.av) * math.cos(self.angle)

    # ------------------------------------------------ #

    def get_pos(self):
        
        x = (int)(self.x % (WIN_X - 1))
        y = (int)(self.y % (WIN_Y - 1))
        return(x, y, self.angle)
    
    def set_activations(self, action):
        self.av = action[0]
        self.turn = action[1] - action[2]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def _activation_av(av):
    if av > 0.5:
        return(1)
    return(0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def _activation_turn(turn):
    if turn > 0.5:
        return((turn - 0.5) * 2 * math.pi / 2.0)
    elif turn < -0.5:
        return((turn + 0.5) * 2 * math.pi / 2.0)
    return(0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def example():

    random.seed(5)
    a = Stagevars()
    a.av = 1
    for i in range(10):
        a.action()
        a.turn = (random.random()-0.5)*2.0
        print(a.angle)
        print(a.x)
        print(a.y)


