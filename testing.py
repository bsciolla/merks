
import stagevars
import numpy
import random
import math
import matplotlib as plt
import matplotlib.pyplot as plt

from globalvars import WIN_X
from globalvars import WIN_Y
from globalvars import NB_AREAS
from globalvars import PATCHSIZE

import genome
genome.test_genome()

import neuralnetwork
neuralnetwork.test_satisfies()
neuralnetwork.test_find_best_match()
neuralnetwork.test_sort_sensors_actions()
neuralnetwork.test_create_index_but()
neuralnetwork.test_Neuralnetwork()
neuralnetwork.test_Neuralnetwork_create_neuron_from_model()
neuralnetwork.test_permute_connectivity()
neuralnetwork.test_get_idx_list_for_sensors()
neuralnetwork.test_double_fancy_indexing()

import rule
rule.test_readsegment()
rule.test_readrule()

import nnbuilder
nnbuilder.test_build_nn()
