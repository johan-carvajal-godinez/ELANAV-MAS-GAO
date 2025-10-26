import random as rd
import VisualizationGA as Viz
import numpy as np
import networkx as nx
from timeit import default_timer as timer
import matplotlib.pyplot as plt

########################################################################################################################
# Parameters for Simulation
NUM_AGENTS = 12               # Number of Agents allocated to the Software Architecture
NUMB_OF_ELITE_CHROMOSOMES = int(NUM_AGENTS/5)
TOURNAMENT_SELECTION_SIZE = int(NUM_AGENTS/3)
# MUTATION_RATE = 1/(NUM_AGENTS*0.5*(NUM_AGENTS-1))
MUTATION_RATE = 1/NUM_AGENTS
print(MUTATION_RATE)
COST_MAT = np.triu(np.ones((NUM_AGENTS, NUM_AGENTS)))  # Parameters for Cost
POPULATION_SIZE = 15  # Number of Topology Candidates
HIERARCHY1 = [3, 4, 5]        # Hierarchical Constraints
HIERARCHY2 = [6, 7, 8]        # Hierarchical Constraints
TEAM1 = [2, 3, 6]
CASE_2_H = [[3, 4, 5], [6, 7, 8]]
CASE_2_T = [[2, 3, 6]]
GENERATION = 1
TEAMS = [TEAM1]
HIERARCHIES = [HIERARCHY1, HIERARCHY2]


########################################################################################################################
class Chromosome:
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
        self._is_golden = False
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

    def get_len(self): return self._genes.__len__()

    def is_coordinator(self, node):  # This function return true if the specified node belong to the coordinator list
        is_coordinator = False
        coordinators = sorted(self.get_master_nodes())
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
        self._target_cost = 4*self._num_agents-self.get_fixed_cost()
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

    def is_golden_gen(self, gen):
        self._is_golden = False
        if self._golden_genes[gen] == 1:
            self._is_golden = True
        return self._is_golden

    def set_cost_mat(self, cost_mat):  # Setter for cost matrix
        self._cost_par = cost_mat

    def get_fitness(self):
        self._fitness = 0
        functional = self.get_functional_nodes()
        degs = list(self.get_graph().degree().values())
        #print(degs)
        deg_list = []
        for x in functional:
            if degs[x-1] >= 3:
                if degs[x-1] <= 3+int(self._num_agents/10):
                    deg_list.append(1)
                else:
                    deg_list.append(0)
            else:
                deg_list.append(0)
        summa = 0
        # print(deg_list)
        for t in range(deg_list.__len__()):
            if deg_list[t] == 1:
                summa += 1
        if self.test_degree_coord():
            self._fitness = summa/(deg_list.__len__())
        return self._fitness

    '''
    def get_fitness(self):             # Getter for Architecture's fitness
        self._fitness = 0
        self._fitness = (self.get_genes().__len__()-self.get_total_cost()+self.get_fixed_cost())/self.get_genes().__len__()
        return self._fitness
    '''
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
        deg = list(self.get_graph().degree().values())
        deg_coord = []
        for x in coordinators:
            if deg[x-1] >= 3:
                if deg[x-1] <= (((self._num_agents-1)/2)+1):
                    deg_coord.append(1)
                else:
                    deg_coord.append(0)
            else:
                deg_coord.append(0)
        summ = 0
        for t in range(deg_coord.__len__()):
            if deg_coord[t] == 1:
                summ += 1
        if summ == deg_coord.__len__():
            self._degree_coord = True
        return self._degree_coord

    def test_degree_funct(self):
        self._degree_funct = False
        functional = self.get_functional_nodes()
        deg2 = list(self.get_graph().degree().values())
        deg_funct = []
        for x in functional:
            if deg2[x-1] == 4:
                deg_funct.append(1)
            else:
                deg_funct.append(0)
        summ2 = 0
        for t in range(deg_funct.__len__()):
            if deg_funct[t] == 1:
                summ2 += 1
        if summ2 == deg_funct.__len__():
            self._degree_funct = True
        return self._degree_funct

    def __str__(self):  # Overwrite the string method in the genes object
        return self._genes.__str__()


########################################################################################################################
class Population:
    def __init__(self, num_agents, pop_size, cost_par_mat, team_int_constraints, hierarchical_int_constraints):
        self._chromosomes = []
        self._pop_size = pop_size
        i = 0
        trial = 0
        while i < pop_size:
            candidate_crom = Chromosome(num_agents, cost_par_mat)
            # Set All the Hierarchy Groups
            k = 0
            while k < hierarchical_int_constraints.__len__():
                candidate_crom.set_constraint_org_hierarchy(hierarchical_int_constraints[k])
                k += 1
            k = 0
            # Set All the Team Groups
            while k < team_int_constraints.__len__():
                candidate_crom.set_constraint_org_team(team_int_constraints[k])
                k += 1
            # Append Chromosome to Candidate List
            self._chromosomes.append(candidate_crom)
            i += 1

    def get_chromosomes(self):  # Getter for a list of candidate MAS-based Architectures
        return self._chromosomes

    def get_population_size(self):
        return self._pop_size


########################################################################################################################
class GeneticAlgorithm:
    @staticmethod
    def _crossover_population(pop):
        np.fill_diagonal(COST_MAT, 0)
        crossover_pop = Population(NUM_AGENTS, 0, COST_MAT, TEAMS, HIERARCHIES)
        for i in range(NUMB_OF_ELITE_CHROMOSOMES):
            crossover_pop.get_chromosomes().append(pop.get_chromosomes()[i])
        i = NUMB_OF_ELITE_CHROMOSOMES
        while i < POPULATION_SIZE:
            chromosome1 = GeneticAlgorithm._select_tournament_population(pop).get_chromosomes()[0]
            chromosome2 = GeneticAlgorithm._select_tournament_population(pop).get_chromosomes()[0]
            crossover_pop.get_chromosomes().append(GeneticAlgorithm._crossover_chromosomes(chromosome1, chromosome2))
            i += 1
        return crossover_pop

    @staticmethod
    def _mutate_population(pop):
        for i in range(NUMB_OF_ELITE_CHROMOSOMES, POPULATION_SIZE):
            GeneticAlgorithm._mutate_chromosome(pop.get_chromosomes()[i])
        return pop

    @staticmethod
    def evolve(pop):
        return GeneticAlgorithm._mutate_population(GeneticAlgorithm._crossover_population(pop))

    @staticmethod
    def _crossover_chromosomes(chromosome1, chromosome2):
        crossover_chrom = Population(NUM_AGENTS, 1, COST_MAT, TEAMS, HIERARCHIES).get_chromosomes()[0]
        for i in range(crossover_chrom.get_len()):
            if rd.random() >= 0.5:
                crossover_chrom.get_genes()[i] = chromosome1.get_genes()[i]
            else:
                crossover_chrom.get_genes()[i] = chromosome2.get_genes()[i]
        return crossover_chrom

    @staticmethod
    def _mutate_chromosome(chromosome):
        for i in range(chromosome.get_len()):
            if not chromosome.is_golden_gen(i):
                if rd.random() < MUTATION_RATE:
                    if rd.random() < 0.5:
                        chromosome.get_genes()[i] = 1
                    else:
                        chromosome.get_genes()[i] = 0

    @staticmethod
    def _select_tournament_population(pop):
        tournament_pop = Population(NUM_AGENTS, 0, COST_MAT, TEAMS, HIERARCHIES)
        i = 0
        while i < TOURNAMENT_SELECTION_SIZE:
            tournament_pop.get_chromosomes().append(pop.get_chromosomes()[rd.randrange(0, POPULATION_SIZE)])
            i += 1
        tournament_pop.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
        return tournament_pop


########################################################################################################################
def _print_population(pop, gen_number): # Function to print the current population performance
    print("Interaction Constraints: ", pop.get_chromosomes()[0].get_constraints())
    print("Coordination Nodes: ", sorted(pop.get_chromosomes()[0].get_master_nodes()))
    print("\n-----------------------------------------------------------------------")
    print("Generation #", gen_number, "| Fittest chromosome fitness:", round(pop.get_chromosomes()[0].get_fitness(), 3))
    print("Number of Agents: ", NUM_AGENTS)
    print("Max Communication Cost is: ", pop.get_chromosomes()[0].get_len())
    print("Fixed Communication Cost is: ", pop.get_chromosomes()[0].get_fixed_cost())
    print("Min Communication Cost is: ", pop.get_chromosomes()[0].get_total_cost())
    print("-----------------------------------------------------------------------")
    i = 0
    for x in pop.get_chromosomes():
        print("Chromosome #", i, " :", x, "| Fitness : ", round(x.get_fitness(), 3), "| Total Cost : ", x.get_total_cost(), "| Fix_Cost : ", x.get_fixed_cost())
        i += 1

########################################################################################################################


# Main Program
start = timer()
population = Population(NUM_AGENTS, POPULATION_SIZE, COST_MAT, CASE_2_T, CASE_2_H)
population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
print("Initial Population")
_print_population(population, 0)
generation_number = 1
iterations = [1]
cum_fitness = [0]
total_cost = [None]


while population.get_chromosomes()[0].get_fitness() < 1:
    population = GeneticAlgorithm.evolve(population)
    population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
    _print_population(population, generation_number)
    iterations.append(generation_number)
    cum_fitness.append(population.get_chromosomes()[0].get_fitness())
    total_cost.append(population.get_chromosomes()[0].get_total_cost())
    generation_number += 1
end = timer()
print("Execution Time: "+str(end-start)+" seconds")
topologies = Viz.VisualizationGA(NUM_AGENTS)
candidate_top = population.get_chromosomes()[0].get_genes()
interact = sorted(population.get_chromosomes()[0].get_master_nodes())
team_edges = sorted(population.get_chromosomes()[0].get_team_edges())
h_edges = sorted(population.get_chromosomes()[0].get_hierarchy_edges())
topologies.visualize_topology(candidate_top, interact, "Candidate_Arch_GA"+str(NUM_AGENTS)+"_", GENERATION, team_edges, h_edges)

fig = plt.figure()
plt.plot(iterations, cum_fitness)
fig.suptitle('Fitness Evolution', fontsize=18)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Cumulative Fitness', fontsize=16)
fig.savefig("Fitness_Evolution_"+str(NUM_AGENTS)+"_"+str(GENERATION))

fig2 = plt.figure()
plt.plot(iterations, total_cost)
fig2.suptitle('Total Cost of Communication', fontsize=18)
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Total Cost', fontsize=16)
fig2.savefig("Cost_Evolution_"+str(NUM_AGENTS)+"_"+str(GENERATION))


########################################################################################################################
# The End!
