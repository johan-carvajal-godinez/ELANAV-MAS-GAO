import random as rd
import numpy as np
import networkx as nx


class Chromosome:   # Class For Creating Topologies for MAS-based Architectures
    def __init__(self, n, cost_par_mat):
        self._genes = []
        self._golden_genes = []
        self._graph = nx.Graph(name="MAS-ARCH")
        self._valid = 0
        self._fitness = 0
        self._cost_par = cost_par_mat
        self._target_cost = 0
        self._cost_tot = 0
        self._A = np.zeros((n, n))
        self._num_agents = n
        self._adj_list = []
        self._master_nodes = []
        self._functional_nodes = []
        self._team_edges = []
        self._hierarchy_edges = []
        self._int_constraints = []
        self._fixed_cost = 0
        self._coherence = False
        self._degree_coord = False
        self._degree_funct = False
        self._clustering_array = [n-1]
        self._degree_range = [3, (round((n-1)/3)+2)]
        i = 0
        while i < (self._num_agents*(self._num_agents-1)/2):
            if i < (self._num_agents-1):
                self._genes.append(1)
                self._golden_genes.append(1)
            else:
                if rd.random() >= 0.5:
                    self._genes.append(1)
                    self._golden_genes.append(0)
                else:
                    self._genes.append(0)
                    self._golden_genes.append(0)
            i += 1

    def is_coordinator(self, node):  # This function return true if the specified node belong to the coordinator list
        is_coordinator = False
        coordinators = sorted(self.get_master_nodes())
        # print(coordinators)
        i = 0
        while i < self.get_master_nodes().__len__():
            if node == coordinators[i]:
                is_coordinator = True
            i += 1
        return is_coordinator

    def get_coherence(self):  # Coherence is TRUE if the degree of a candidate architecture is within the pre-defined degree range
        self._coherence = False
        degrees = list(self.get_graph().degree().values())
        degrees.pop(0)  # Remove AMS interactions
        if min(degrees) >= self._degree_range[0]:
            if max(degrees) <= self._degree_range[1]:
                self._coherence = True
            else:
                self._coherence = False
        return self._coherence

    def get_degree_range(self):
        return self._degree_range

    def set_gen(self, gen):
        self._genes[gen] = 1

    def get_target_cost(self):
        self._target_cost = round(self.get_fixed_cost()+(((self._num_agents*(self._num_agents-1)/2)-self.get_fixed_cost())/2))
        return self._target_cost

    def get_cost_par_mat(self):
        return self._cost_par

    def get_adj_mat(self):  # Get the adjacency matrix for a given top
        i = 1
        while i <= (self._num_agents-1):
            j = i+1
            while j <= self._num_agents:
                index = (2*self._num_agents-i)*(i-1)/2+j-i
                self._A[i-1][j-1] = self._genes[round(index-1)]
                j += 1
            i += 1
        return self._A

    def get_total_cost(self):  # Get the total cost of a topology given a cost matrix parameter
        self._cost_tot = 0
        ad_mat = self.get_adj_mat()
        i = 1
        while i <= (self._num_agents-1):
            j = i+1
            while j <= self._num_agents:
                self._cost_tot += self._cost_par[i-1][j-1]*ad_mat[i-1][j-1]
                j += 1
            i += 1
        return self._cost_tot

    def get_fixed_cost(self):
        self._fixed_cost = 0
        for i in range(self._golden_genes.__len__()):
            if self._golden_genes[i] == 1:
                self._fixed_cost += 1
        return self._fixed_cost

    def get_genes(self):
        return self._genes

    def get_golden_genes(self):
        return self._golden_genes

    def set_cost_mat(self, cost_mat):  # Setter for cost matrix
        self._cost_par = cost_mat

    def get_fitness(self):
        self._fitness = 0
        functional = self.get_functional_nodes()
        degs = list(self.get_graph().degree().values())
        # print(degs)
        deg_list = []
        for x in functional:
            if degs[x-1] == 3:
                deg_list.append(1)
            else:
                deg_list.append(0)
        summa = 0
        for t in range(deg_list.__len__()):
            if deg_list[t] == 1:
                summa += 1
        if self.test_degree_coord():
            self._fitness = summa/(deg_list.__len__())
        return self._fitness

    def get_graph(self):  # Get the graph structure for a candidate topology specified by a set of genes
        self._adj_list = []
        i = 1
        self._graph.add_node(i)
        while i <= (self._num_agents-1):
            row = []
            j = i+1
            while j <= self._num_agents:
                index = (2*self._num_agents-i)*(i-1)/2+j-i
                index = index-1
                row.append(self._genes[round(index)])
                if self._genes[round(index)] == 1:
                    self._graph.add_edge(i, j)
                j += 1
                self._adj_list.append(row)
            i += 1
        self._graph.add_node(i)
        return self._graph

    def get_adj_list(self):
        return self._adj_list

    def set_constraint_org_team(self, node_list):  # Set an team interaction constraint  specified by a node list
        self._master_nodes.append(node_list[0])
        self._int_constraints.append(node_list)
        i = 1
        while i <= (node_list.__len__()-1):
            index = (2*self._num_agents-node_list[0])*(node_list[0]-1)/2+node_list[i]-node_list[0]
            self._genes[round(index-1)] = 1
            self._golden_genes[round(index-1)] = 1
            self._team_edges.append([node_list[0], node_list[i]])
            i += 1

    def get_team_edges(self):
        return self._team_edges

    def get_constraints(self):  # Return all the interaction defined in a candidate topology
        return self._int_constraints

    def get_master_nodes(self):         # Get the list of Master Nodes in the Topology
        return self._master_nodes

    def set_constraint_org_hierarchy(self, node_list):  # Set an hierarchical interaction constraint specified by a node list
        self._master_nodes.append(node_list[0])
        self._int_constraints.append(node_list)
        i = 1
        while i < (node_list.__len__()):
            index = (2*self._num_agents-node_list[0])*(node_list[0]-1)/2+node_list[i]-node_list[0]
            self._genes[round(index-1)] = 1
            self._golden_genes[round(index-1)] = 1
            self._hierarchy_edges.append([node_list[0], node_list[i]])
            j = i + 1
            while j < (node_list.__len__()):
                index2 = (2*self._num_agents-node_list[i])*(node_list[i]-1)/2+node_list[j]-node_list[i]
                self._genes[round(index2-1)] = 0
                self._golden_genes[round(index-1)] = 1
                j += 1
            i += 1

    def get_hierarchy_edges(self):
        return self._hierarchy_edges

    def get_functional_nodes(self):
        self._functional_nodes = []
        j = 2
        while j <= self._num_agents:
            if not self.is_coordinator(j):
                self._functional_nodes.append(j)
            j += 1
        return self._functional_nodes

    def test_degree_coord(self):
        self._degree_coord = False
        coordinators = self.get_master_nodes()
        # functional = self.get_functional_nodes()
        # print(coordinators, functional)
        deg = list(self.get_graph().degree().values())
        # print(deg)
        deg_coord = []
        for x in coordinators:
            # print(x, deg[x-1])
            if deg[x-1] >= 3:
                if deg[x-1] <= (((self._num_agents-1)/2)+1):
                    # print("Const Sat")
                    deg_coord.append(1)
                else:
                    deg_coord.append(0)
            else:
                # print("Const UnSat")
                deg_coord.append(0)
        # print(deg_coord)
        summ = 0
        for t in range(deg_coord.__len__()):
            if deg_coord[t] == 1:
                summ += 1
        # print(summ, deg_coord.__len__())
        if summ == deg_coord.__len__():
            self._degree_coord = True
        return self._degree_coord

    def test_degree_funct(self):
        self._degree_funct = False
        functional = self.get_functional_nodes()
        deg = list(self.get_graph().degree().values())
        deg_funct = []
        for x in functional:
           for x in functional:
            if deg[x-1] >= 3:
                if deg[x-1] <= 3+int(self._num_agents/10):
                    deg_funct.append(1)
                else:
                    deg_funct.append(0)
            else:
                deg_funct.append(0)
        summ = 0
        for t in range(deg_funct.__len__()):
            if deg_funct[t] == 1:
                summ += 1
        if summ == deg_funct.__len__():
            self._degree_funct = True
        return self._degree_funct

    def __str__(self):  # Overwrite the string method in the genes object
        return self._genes.__str__()
