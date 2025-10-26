import networkx as nx
import matplotlib.pyplot as plt


class VisualizationGA:
    def __init__(self, size):
        self._n = size
        self._adj_list = []

    def visualize_topology(self, architecture, interactions, figure_name, figure_number, team_edges, hier_edges):
        graph = nx.Graph(name="MAS-ARCH")
        i = 1
        # Construct the Graph based on the topology architecture provided as input parameter
        graph.add_node(i)
        color_array = ['r']
        node_size = [1000]
        while i <= (self._n-1):
            color_array.append('g')
            node_size.append(400)
            row = []
            j = i+1
            while j <= self._n:
                index = (2*self._n-i)*(i-1)/2+j-i
                index = index-1
                row.append(architecture[round(index)])
                if architecture[round(index)] == 1:
                    if i == 1:
                        graph.add_edge(i, j, stylo='dashed', weight=0.5)
                    else:
                        graph.add_edge(i, j, stylo='dotted', weight=2)
                j += 1
                self._adj_list.append(row)
            i += 1
        graph.add_node(i)
        # Change the property of the Master Nodes of the Fixed Interactions Constraints
        for x in interactions:
            color_array[x-1] = 'c'
            node_size[x-1] = 600
        # Change the properties of the edges for the Fixed Interactions Constraints
        k = 0
        while k <= (team_edges.__len__()-1):
            t = 0
            while t < team_edges[k].__len__()-1:
                graph.add_edge(team_edges[k][t], team_edges[k][t+1], stylo='solid', weight=1)
                t += 1
            k += 1
        k = 0
        while k <= (hier_edges.__len__() - 1):
            t = 0
            while t < hier_edges[k].__len__() - 1:
                graph.add_edge(hier_edges[k][t], hier_edges[k][t + 1], stylo='solid', weight=2)
                t += 1
            k += 1
        # Visualize the Graph and store it for further analysis
        edges = graph.edges()
        edge_style = [graph[u][v]['stylo'] for u, v in edges]
        weights = [graph[u][v]['weight'] for u, v in edges]
        plt.clf()
        nx.draw_networkx(graph, pos=nx.shell_layout(graph), style=edge_style, width=weights, with_labels=True, hold=False, node_color=color_array, node_size=node_size)
        plt.draw()
        plt.axis('off')
        plt.savefig(figure_name+str(figure_number)+".png")  # save as png
        # plt.show()
        print(list(graph.degree().values()))
        print(list(nx.shortest_path_length(graph, source=2).values()))
        plt.clf()
