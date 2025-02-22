from typing import List
import stage
import merk
import GrowNeuralNetwork

import numpy
import random
import math

from globalvars import AGEING_RATE, NUM_RECYCLE, ADN_LENGTH

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def aging(mek: merk.Merk, stage_: stage.Stage) -> None:
    field = stage_.get_local_cost(mek.merkState.x, mek.merkState.y)
    delta = 0
    if abs(mek.merkState.turn) > 0.5:
        delta += 0.25
    if mek.merkState.moving > 0.5:
        delta += 0.25
    
    mek.merkState.health += delta * AGEING_RATE
    
    # if mek.svars.av <= 0.1:
        #mek.svars.health = mek.svars.health - AGEING_RATE/2.0
    # if mek.svars.turn > -0.3:
        #mek.svars.health = mek.svars.health - AGEING_RATE/2.0


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def deathandrecycle(merklist: List[merk.Merk]) -> None:
    healthlist = numpy.array([mek.merkState.health for mek in merklist])
    healthsort = healthlist.argsort()
    for i in range(0, NUM_RECYCLE):
        idx_die1 = healthsort[i]
        idx_rep1 = random.randint(0, healthsort.size-1)
        rank = random.randint(1, 15)
        idx_rep2 = healthsort[-rank]
        mek = newmerk(merklist[idx_rep1], merklist[idx_rep2])
        merklist[idx_die1] = mek


def newmerk(merk1: merk.Merk, merk2: merk.Merk) -> merk.Merk:
    mek = merk.Merk()
    isec = random.randint(0, ADN_LENGTH-1)
    mek.genome.adn = merk1.genome.adn[:isec] + merk2.genome.adn[isec:]

    iran1 = random.randint(0, ADN_LENGTH-1)
    iran2 = random.randint(0, ADN_LENGTH-1)
    if iran1 > iran2:
        i = iran1
        iran1 = iran2
        iran2 = i
    if iran2-iran1 > 5:
        iran2 = iran1 + 5
    for i in range(iran1, iran2):
        mek.genome.adn[i] = random.randint(0, 1)

    for j in range(10):
        mek.genome.adn[random.randint(0, ADN_LENGTH-1)] = random.randint(0, 1)

    mek.genome.make_clean_rules()
    GrowNeuralNetwork.GrowNeuralNetwork(mek.neuralNetwork, mek.genome, verbose=False)
    return(mek)
