
import matplotlib.pyplot as plt
import networkx as nx
import numpy
plt.ion()


# test
def plot_neural_network(mek):
    G = nx.DiGraph(mek.nn.links)
    mylabels = dict(zip(range(len(mek.nn.neurons)),
                        [to_string(i)+'\n#'
                         + str(ix)+'' for (ix, i) in enumerate(mek.nn.neurons)]))

    G = nx.relabel_nodes(G, mylabels)
    pos = nx.layout.spring_layout(G, k=2)

    epos = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.5]
    eneg = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= -0.5]

    arrowsize = 50
    colorspos = numpy.arange(len(epos))/5.0+4.0*len(epos)/5.0
    colorsneg = numpy.arange(len(eneg))/5.0+4.0*len(eneg)/5.0
    nx.draw_networkx_edges(G, pos, edgelist=epos, edge_color=colorspos,
                           width=3, arrowsize=arrowsize, alpha=1, arrowstyle='->', edge_cmap=plt.cm.Blues)
    nx.draw_networkx_edges(G, pos, edgelist=eneg,
                           width=2, arrowsize=arrowsize, alpha=1, edge_color=colorsneg, arrowstyle='->', edge_cmap=plt.cm.Reds)

    nodes = nx.draw_networkx_nodes(
        G, pos, node_size=1500, node_color='gray', alpha=1)
    nx.draw_networkx_labels(G, pos, font_size=10,
                            font_family='sans-serif', font_weight='bold')
    ax = plt.gca()
    ax.set_axis_off()
    plt.show()


def to_string(name):
    out = ""
    for i in name:
        out = out + str(i)
    return(out)
