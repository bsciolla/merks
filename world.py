from typing import List
import numpy as np
import stage
import merk
import lifeanddeath

import random
import math
import matplotlib as plt
import matplotlib.pyplot as plt

from globalvars import WIN_X
from globalvars import WIN_Y
from globalvars import NB_AREAS
from globalvars import PATCHSIZE
from MerkState import Activation

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class World:

    # ------------------------------------------------ #

    def __init__(self, merks_num: int = 0) -> None:
        self.stage_: stage.Stage = stage.Stage()
        self.merklist: List[merk.Merk] = []
        self.merks_num: int = merks_num
        for _ in range(merks_num):
            new_merk = merk.Merk()
            new_merk.build_random_merk()
            self.merklist.append(new_merk)

    # ------------------------------------------------ #

    def step_forward(self) -> None:
        self.stage_.move_areas()
        self.stage_.refresh_costmap()
        for merk in self.merklist:
            actions = update_activations(merk, self.stage_)
            merk.merkState.set_activations(actions)
            merk.action()
            lifeanddeath.aging(merk, self.stage_)
        lifeanddeath.deathandrecycle(self.merklist)

    # ------------------------------------------------ #


def update_sprites(spritelist: List, world: 'World') -> None:
    for ix, merk_sprite in enumerate(spritelist):
        x, y, angle = world.merklist[ix].merkState.get_pos()
        merk_sprite.update(x, y, angle)


def update_activations(mek: merk.Merk, stage_: stage.Stage) -> Activation:
    x, y, angle = mek.merkState.x, mek.merkState.y, mek.merkState.angle
    sensorsdata = np.array(
        list(stage_.get_local_fields(x, y, angle)) + [1.0, -1.0])
    
    return mek.neuralNetwork.update_activations(sensorsdata)
