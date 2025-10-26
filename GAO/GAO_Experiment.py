import random as rd
import VisualizationGA as Viz
import numpy as np
import networkx as nx
from timeit import default_timer as timer
import matplotlib.pyplot as plt

########################################################################################################################
# Parameters for Simulation
NUM_AGENTS = 12                                        # Number of Agents allocated to the Software Architecture
COST_MAT = np.triu(np.ones((NUM_AGENTS, NUM_AGENTS)))  # Parameters for cost of communication between agents C_i

# Interaction for nominal 10
# HIERARCHIES = [[2, 8, 9, 10], [3, 4, 5]]               # Hierarchical interactions that constrain the solution
# TEAMS = [[4, 8, 9], [5, 10]]                           # Team interactions that constraint the solution

# Interaction for nominal 20 and 30
# HIERARCHIES = [[2, 8, 9, 10, 11, 12, 13, 14, 15], [3, 4, 5, 18, 19]]               # Hierarchical interactions that constrain the solution
#TEAMS = [[4, 8, 9, 11, 12], [5, 10, 12, 17]]                           # Team interactions that constraint the solution

# Interaction for min
HIERARCHIES = [[2, 8]]                                  # Hierarchical interactions that constrain the solution
TEAMS = [[5, 10]]                                      # Team interactions that constraint the solution

STATIC = False                                         # Flag to define the GAO implementation parameters (0=dynamic)
REP = 30                                               # Number of trials for the simulation
FNAME_FITNESS_EVOL = "FitnessEvolution.csv"            # File name for storing the fitness evolution results
FNAME_TIME= "Time_Taken.csv"                           # File name for storing the fitness evolution results
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
        # Initialize the value
        self._fitness = 0

        # Get a list of functional nodes in the candidate architecture
        functional = self.get_functional_nodes()

        # Obtains the degrees of all nodes in the candidate architecture
        degs = list(self.get_graph().degree().values())

        # Initialize an empty list to store the results of the flexible constrain tests
        deg_list = []

        # Check that the candidate architecture satisfies the flexible constrains in the functional nodes
        for x in functional:
            if degs[x-1] >= 3:
                if degs[x-1] <= 3+int(self._num_agents/10):
                    deg_list.append(1)
                else:
                    deg_list.append(0)
            else:
                deg_list.append(0)

        # Initialize a temporal variable to store the sum of nodes that satisfy the flexible constrains
        summa = 0

        # sum up all the nodes of the candidate architecture that satisfy the functional nodes interaction constrains
        for t in range(deg_list.__len__()):
            if deg_list[t] == 1:
                summa += 1

        # Verify that the candidate architecture satisfies the coordination nodes constrains, if so returns the fitness
        if self.test_degree_coord():
            self._fitness = summa/(deg_list.__len__())
        # Return the fitness value
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


# Main Program for Topological Optimization of Software Architectures based on a Genetic Algorithm

# Initialize Temporal Variables

trial_number = 1
# GAO Initial Parameters

NUMB_OF_ELITE_CHROMOSOMES = 4
#NUMB_OF_ELITE_CHROMOSOMES = int(NUM_AGENTS/4)

TOURNAMENT_SELECTION_SIZE = 8
# TOURNAMENT_SELECTION_SIZE = int(NUM_AGENTS/2)

MUTATION_RATE = 1/(0.5*NUM_AGENTS*(NUM_AGENTS-1))

POPULATION_SIZE = 10  # Number of Topology Candidates per generation
# POPULATION_SIZE = NUM_AGENTS  # Number of Topology Candidates per generation


while trial_number <= REP:
    iterations = [1]
    cum_fitness = [0]
    total_cost = [None]
    generation_number = 1
    # Start time counter
    start = timer()

    # Create and sort a random population with fixed interaction constraints
    population = Population(NUM_AGENTS, POPULATION_SIZE, COST_MAT, TEAMS, HIERARCHIES)
    population.get_chromosomes().sort(key=lambda x: x.get_total_cost(), reverse=False)
    population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
    print("Initial Population")
    _print_population(population, 0)

    # Check the optimization flag, to determine if the mutation rate is static or dynamic
    if STATIC:
        while population.get_chromosomes()[0].get_fitness() < 1:
            # Evolve the current population
            population = GeneticAlgorithm.evolve(population)

            # Sort the generated population by cost and fitness for the next iteration
            population.get_chromosomes().sort(key=lambda x: x.get_total_cost(), reverse=False)
            population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
            _print_population(population, generation_number)

            # Update variables for the figures
            iterations.append(generation_number)
            cum_fitness.append(population.get_chromosomes()[0].get_fitness())
            total_cost.append(population.get_chromosomes()[0].get_total_cost())

            # Update generation number
            generation_number += 1
        end = timer()
    else:
        # Initialize local variables to adjust the mutation rate
        local_acum = 0
        last_cost = population.get_chromosomes()[0].get_total_cost()

        while population.get_chromosomes()[0].get_fitness() < 1:
            # Verify if the simulation there is in a local minimum
            if last_cost == population.get_chromosomes()[0].get_total_cost():
                local_acum += 1
            else:
                local_acum = 0
            print(local_acum)
            # Update the last cost
            last_cost = population.get_chromosomes()[0].get_total_cost()

            # Change the mutation rate to improve exploration when solution finds a local optima
            if local_acum > 100:
                if MUTATION_RATE < 0.9:
                    MUTATION_RATE = MUTATION_RATE * 1.001
                else:
                    MUTATION_RATE = 0.1
            else:
                MUTATION_RATE = 1 / (0.5 * NUM_AGENTS * (NUM_AGENTS - 1))

            # Evolve the population according the GAO parameters
            population = GeneticAlgorithm.evolve(population)

            # Sort and print the generated population by cost and fitness for the next iteration
            population.get_chromosomes().sort(key=lambda x: x.get_total_cost(), reverse=False)
            population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
            _print_population(population, generation_number)

            # Update variables for the figures
            fe = open(FNAME_FITNESS_EVOL, "a+")
            fe.write(str(trial_number) + "," + str(generation_number) + "," + str(
                population.get_chromosomes()[0].get_fitness()) + "," + str(
                population.get_chromosomes()[0].get_total_cost()) + "\n")
            fe.close()
            iterations.append(generation_number)
            cum_fitness.append(population.get_chromosomes()[0].get_fitness())
            total_cost.append(population.get_chromosomes()[0].get_total_cost())

            # Update generation number
            generation_number += 1
        end = timer()
        ft = open(FNAME_TIME, "a+")
        ft.write(str(trial_number) + "," + str(end - start) + "\n")
        ft.close()

    print("Execution Time: " + str(end - start) + " seconds")

    # Visualize Solution Found
    topologies = Viz.VisualizationGA(NUM_AGENTS)
    candidate_top = population.get_chromosomes()[0].get_genes()
    interact = sorted(population.get_chromosomes()[0].get_master_nodes())
    team_edges = sorted(population.get_chromosomes()[0].get_team_edges())
    h_edges = sorted(population.get_chromosomes()[0].get_hierarchy_edges())
    topologies.visualize_topology(candidate_top, interact, "Candidate_Arch_GA" + str(NUM_AGENTS) + "_", trial_number, team_edges,
                                  h_edges)

    # Generates a figure for fitness evolution
    fig = plt.figure()
    plt.plot(iterations, cum_fitness)
    fig.suptitle('Fitness Evolution', fontsize=18)
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Cumulative Fitness', fontsize=16)
    fig.savefig("Fitness_Evolution_" + str(NUM_AGENTS) + "_" + str(trial_number))

    # Generates a figure for total cost evolution
    fig2 = plt.figure()
    plt.plot(iterations, total_cost)
    fig2.suptitle('Total Cost of Communication', fontsize=18)
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Total Cost', fontsize=16)
    fig2.savefig("Cost_Evolution_" + str(NUM_AGENTS) + "_" + str(trial_number))

    trial_number += 1

########################################################################################################################
# The End!
