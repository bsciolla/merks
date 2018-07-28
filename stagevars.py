
import random
import math

from globalvars import WIN_X, WIN_Y, GRID_X, GRID_Y

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class Stagevars:

    # ------------------------------------------------ #

    def __init__(self):

        # general implicit
        self.x = random.randint(0, GRID_X)
        self.y = random.randint(0, GRID_Y)
        self.angle = random.randint(0, 3)
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
        self.angle = round(self.angle + self.turn) % 4
        if self.av > 0:
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

    def get_pos(self):

        x = (int)(self.x % GRID_X)
        y = (int)(self.y % GRID_Y)
        return(x, y, self.angle)

    def set_activations(self, action):
        self.av = round(action[0])
        self.turn = round(action[1]) - round(action[2])


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
