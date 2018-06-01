
from globalvars import SENSORS, ACTIONS
from plot_graphs import plot_neural_network
# Utility functions to analyze the current species


interest = []
less = []
ml = thisworld.merklist

# Find most sensible nodes
for ix, mek in enumerate(ml):

    nsensors = numpy.array(mek.nn.sensors_found).sum()
    nactions = numpy.array(mek.nn.actions_found).sum()
    print('\n  ', ix)
    print("neurons: ",
          mek.nn.nb_neurons)
    print("sensors: ", nsensors)
    print("actions: ",
          nactions
          )

    if nsensors == len(SENSORS) and nactions == len(ACTIONS):
        interest.append(ix)

    if nsensors == len(SENSORS) or nactions == len(ACTIONS):
        less.append(ix)


for idx in interest:
    plot_neural_network(ml[idx])


gendist = []
for mek2 in ml:
    gendist.append(proxygen(mek, mek2))

gendist = numpy.array(gendist)
gendist.mean()
gendist[less].mean()


def proxygen(mek, mek2):
    return(
        numpy.abs(numpy.array(mek.gen.adn) -
                  numpy.array(mek2.gen.adn)).sum()/len(mer.gen.adn)*100)


for ix in less:
    mek = ml[ix]
    nsensors = numpy.array(mek.nn.sensors_found).sum()
    nactions = numpy.array(mek.nn.actions_found).sum()
    print('\n  ', ix)
    print("neurons: ",
          mek.nn.nb_neurons)
    print("sensors: ", nsensors)
    print("actions: ",
          nactions
          )
