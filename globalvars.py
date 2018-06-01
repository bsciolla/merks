
import weightnode

# genome.py
MAX_BINARY_SIZE_FOR_NEURONS = 6
ADN_LENGTH = 1000
MAX_HASH = weightnode.weightnode(
    [1 for i in range(MAX_BINARY_SIZE_FOR_NEURONS)]) + 1

# nnbuilder.py
BUILDING_CYCLES = 3

# neuralnetwork.py
MAX_NEURONS = 50
NEURAL_NOISE = 0.6

# stage.py
WIN_X = 400
WIN_Y = 200
NB_AREAS = 20
PATCHSIZE = 70

# world.py
MERKS_NUM = 150
CLOCK_FPS = 15

# nnbuilder.py
# costmap_value, costmap_grad_forw, costmap_grad_lat
SENSORS = [[0], [1], [0, 0]]
# move forward, turn left, turn right
ACTIONS = [[0, 1], [1, 0], [1, 1]]

# lifeanddeath.py
AGEING_RATE = 0.01
NUM_RECYCLE = 2

