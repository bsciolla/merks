
import stage
import merk
import nnbuilder

import numpy
import random
import math

from globalvars import AGEING_RATE, NUM_RECYCLE, ADN_LENGTH

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def aging(mek, stage_):
    field = stage_.get_local_cost(mek.svars.x, mek.svars.y)
    if field > 0.5:
        mek.svars.health = mek.svars.health + AGEING_RATE/2.0
    else:
        mek.svars.health = mek.svars.health - AGEING_RATE
    # if mek.svars.av <= 0.1:
        #mek.svars.health = mek.svars.health - AGEING_RATE/2.0
    # if mek.svars.turn > -0.3:
        #mek.svars.health = mek.svars.health - AGEING_RATE/2.0


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def deathandrecycle(merklist):
    healthlist = numpy.array([mek.svars.health for mek in merklist])
    healthsort = healthlist.argsort()
    for i in range(0, NUM_RECYCLE):
        idx_die1 = healthsort[i]
        idx_rep1 = random.randint(0, healthsort.size-1)
        rank = random.randint(1, 15)
        idx_rep2 = healthsort[-rank]
        mek = newmerk(merklist[idx_rep1], merklist[idx_rep2])
        merklist[idx_die1] = mek


def newmerk(merk1, merk2):
    mek = merk.Merk()
    isec = random.randint(0, ADN_LENGTH-1)
    mek.gen.adn = merk1.gen.adn[:isec] + merk2.gen.adn[isec:]

    iran1 = random.randint(0, ADN_LENGTH-1)
    iran2 = random.randint(0, ADN_LENGTH-1)
    if iran1 > iran2:
        i = iran1
        iran1 = iran2
        iran2 = i
    if iran2-iran1 > 5:
        iran2 = iran1 + 5
    for i in range(iran1, iran2):
        mek.gen.adn[i] = random.randint(0, 1)

    for j in range(10):
        mek.gen.adn[random.randint(0, ADN_LENGTH-1)] = random.randint(0, 1)

    mek.gen.make_clean_rules()
    nnbuilder.build_nn(mek.nn, mek.gen, verbose=False)
    return(mek)
