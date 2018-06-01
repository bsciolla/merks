
import stagevars
import numpy
import random
import math
#import matplotlib as plt
#import matplotlib.pyplot as plt

from globalvars import WIN_X
from globalvars import WIN_Y
from globalvars import NB_AREAS
from globalvars import PATCHSIZE

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Stage:

    # ------------------------------------------------ #

    def __init__(self):
        self.costmap = numpy.zeros([WIN_X, WIN_Y])
        self.nbareas = NB_AREAS
        self.areas = numpy.zeros([2, self.nbareas])
        self.patch = _patch(PATCHSIZE, PATCHSIZE)

        # find gradient scale
        self.gradscalex = (self.patch[1:, :]-self.patch[:-1, :]).max()
        self.gradscaley = (self.patch[:, 1:]-self.patch[:, :-1]).max()

        self.random_areas()
        self.refresh_costmap()

    # ------------------------------------------------ #

    def refresh_costmap(self):
        self.costmap = numpy.zeros([WIN_X, WIN_Y])
        for i in range(self.areas.shape[1]):
            xp = int(self.areas[0, i])
            yp = int(self.areas[1, i])
            self.costmap[xp:xp+PATCHSIZE, yp:yp+PATCHSIZE] = \
                self.costmap[xp:xp+PATCHSIZE, yp:yp+PATCHSIZE] \
                + self.patch

    # ------------------------------------------------ #

    def random_areas(self):
        for j in range(self.areas.shape[1]):
            self.areas[0, j] = random.random() * (WIN_X - PATCHSIZE - 1)
            self.areas[1, j] = random.random() * (WIN_Y - PATCHSIZE - 1)

    # ------------------------------------------------ #

    def move_areas(self):
        step = 2
        for j in range(self.areas.shape[1]):
            self.areas[0, j] = self.areas[0, j] + step*(random.random()-0.4)
            self.areas[1, j] = self.areas[1, j] + step*(random.random()-0.4)
            self.areas[0, j] = self.areas[0, j] % (WIN_X - PATCHSIZE - 1)
            self.areas[1, j] = self.areas[1, j] % (WIN_Y - PATCHSIZE - 1)

    # ------------------------------------------------ #

    def get_local_fields(self, x, y, angle):
        x = (int)(x % (WIN_X - 1))
        y = (int)(y % (WIN_Y - 1))

        gradx = (self.costmap[x+1, y] - self.costmap[x, y])/self.gradscalex
        grady = (self.costmap[x, y+1] - self.costmap[x, y])/self.gradscaley
        costmap_value = self.costmap[x, y]
        costmap_grad_forw = gradx * math.sin(angle) \
            - grady * math.cos(angle)
        costmap_grad_lat = gradx * math.cos(angle) \
            + grady * math.sin(angle)

        return(costmap_value, costmap_grad_forw, costmap_grad_lat)

    # ------------------------------------------------ #

    def get_local_cost(self, x, y):
        x = (int)(x % (WIN_X - 1))
        y = (int)(y % (WIN_Y - 1))
        return(self.costmap[x, y])

    # ------------------------------------------------ #

    def displaymap(self):
        costmap = self.costmap > 0.5
        showmap = numpy.zeros([costmap.shape[0],
                               costmap.shape[1],
                               3
                               ])
        costmap = numpy.expand_dims(costmap, 2)
        costmap = (costmap/costmap.max())*255/2.0
        showmap[:, :, 0] = costmap[:, :, 0]*0.2 + 155/2.0
        showmap[:, :, 1] = costmap[:, :, 0] + 155/2.0
        showmap[:, :, 2] = costmap[:, :, 0]*0.5 + 205/2.0
        return(showmap)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def _patch(sizex, sizey):
    patch = numpy.zeros([sizex, sizey])
    sigma = 0.15
    for i in range(sizex):
        for j in range(sizey):
            patch[i, j] = math.exp(
                -(i - (sizex - 1)/2.0)**2.0/(2.0*(sizex*sigma)**2.0)
                - (j - (sizey - 1)/2.0)**2.0/(2.0*(sizey*sigma)**2.0)
            )
    return(patch)


def example():
    random.seed(8)
    a = Stage()
    a.areas
    a.refresh_costmap()
    plt.imshow(a.costmap)
    plt.show(block=False)
