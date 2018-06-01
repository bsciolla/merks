
import stage
import merk
import lifeanddeath

import numpy
import random
import math
import matplotlib as plt
import matplotlib.pyplot as plt

from globalvars import WIN_X
from globalvars import WIN_Y
from globalvars import NB_AREAS
from globalvars import PATCHSIZE

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class World:

    # ------------------------------------------------ #

    def __init__(self, merks_num=0):
        self.stage_ = stage.Stage()
        self.merklist = []
        self.merks_num = merks_num
        for i in range(merks_num):
            self.merklist.append(merk.Merk())
            self.merklist[-1].build_random_merk()

    # ------------------------------------------------ #

    def step_forward(self):
        self.stage_.move_areas()
        self.stage_.refresh_costmap()
        for mek in self.merklist:
            actions = update_activations(mek, self.stage_)
            mek.svars.set_activations(actions)
            mek.action()
            lifeanddeath.aging(mek, self.stage_)
        lifeanddeath.deathandrecycle(self.merklist)


    # ------------------------------------------------ #


def update_sprites(spritelist, world):
    for (ix, merk) in enumerate(spritelist):
        x, y, angle = world.merklist[ix].svars.get_pos()
        merk.update(x, y, angle)
        

def update_activations(mek, stage_):
    x, y, angle = mek.svars.x, mek.svars.y, mek.svars.angle
    sensorsdata = numpy.array(
        stage_.get_local_fields(x, y, angle) )
    actions = mek.nn.update_activations(sensorsdata)
    return(actions)
