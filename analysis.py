
from globalvars import SENSORS, ACTIONS
from plot_graphs import plot_neural_network
import numpy
# Utility functions to analyze the current species


def proxygen(mek, mek2):
    return(
        numpy.abs(numpy.array(mek.gen.adn) -
                  numpy.array(mek2.gen.adn)).sum()/len(mek.gen.adn)*100)


healthlist = numpy.array([mek.svars.health for mek in ml])
healthsort = healthlist.argsort()
healthlist[healthsort]
ranks = healthsort.argsort()

interest = []
less = []
ml = thisworld.merklist

# Find most sensible nodes
for ix, merk in enumerate(ml):

    nsensors = numpy.array(merk.nn.sensors_found).sum()
    nactions = numpy.array(merk.nn.actions_found).sum()
    print('\n  ', ix)
    print("neurons: ",
          merk.nn.nb_neurons)
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
    gendist.append(proxygen(merk, mek2))

gendist = numpy.array(gendist)
gendist.mean()
gendist[less].mean()

for ix in healthsort[-10:]:
    merk = ml[ix]
    nsensors = numpy.array(merk.nn.sensors_found).sum()
    nactions = numpy.array(merk.nn.actions_found).sum()
    print('\n  ', ix)
    print(merk.nn.sensors_found)
    print("neurons: ",
          merk.nn.nb_neurons)
    print("sensors: ", nsensors)
    print("actions: ",
          nactions
          )
