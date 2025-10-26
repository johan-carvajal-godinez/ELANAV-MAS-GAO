import Candidate as Pop
import VisualizationGA as Viz
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt

# Parameters for the simulation (DESIGNER DEFINES THEM based on mission Analysis!)

NUM_AGENTS = 7                  # Initial Number of Interacting Agents
MAX_NUM_AGENTS = 11             # Initial Number of Interacting Agents
POPULATION_SIZE = 5             # Number of Topology Candidates
HIERARCHY1 = [3, 4, 5]        # Hierarchical Constraints
HIERARCHY2 = [6, 7]        # Hierarchical Constraints
TEAM1 = [2, 3, 6]
GENERATION = 0
TEAMS = [TEAM1]
HIERARCHIES = [HIERARCHY1, HIERARCHY2]
TIMES = []
AGENTS = []
COST = []
iteration = NUM_AGENTS

file = open("exec_time.csv", "a")

while iteration <= MAX_NUM_AGENTS:
    start = timer()
    COST_MAT = np.triu(np.ones((iteration, iteration)))  # Parameters for Cost
    print("Number of Agents: ", iteration)
    AGENTS.append(iteration)
    np.fill_diagonal(COST_MAT, 0)
    # Create a Population of Candidate Topologies for MAS-based On-board Software
    population = Pop.Candidates(iteration, POPULATION_SIZE, COST_MAT, TEAMS, HIERARCHIES)
    population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=False)
    end = timer()

    # Visualize the Candidate Topologies
    topologies = Viz.VisualizationGA(iteration)
    i = 0
    while i < POPULATION_SIZE:
        candidate_top = population.get_chromosomes()[i].get_genes()
        interact = population.get_chromosomes()[i].get_master_nodes()
        team_edges = population.get_chromosomes()[i].get_team_edges()
        h_edges = population.get_chromosomes()[i].get_hierarchy_edges()
        topologies.visualize_topology(candidate_top, interact, "Candidate_Arch_"+str(iteration)+"_", i, team_edges, h_edges)
        file.write(str(i)+"; "+str(iteration)+"; "+str(end-start)+"; "+str(population.get_chromosomes()[0].get_fitness())+"\n")
        i += 1
    population.view_population(population, population.get_chromosomes(), GENERATION)
    # print("Budget for Total Communication Cost: ", population.get_chromosomes()[0].get_target_cost())
    print("\nElapsed Time", end-start, "seconds")
    TIMES.append(end-start)
    COST.append(population.get_chromosomes()[0].get_fitness())
    # file.write(str(iteration)+"; "+str(end-start)+"; "+str(population.get_chromosomes()[0].get_fitness())+"\n")
    iteration += 1
file.close()
print(TIMES)
print(AGENTS)
print(COST)
fig = plt.figure()
plt.plot(AGENTS, TIMES)
fig.suptitle('Time to Find a Solution with Brute Force', fontsize=18)
plt.xlabel('Number of Agents', fontsize=14)
plt.ylabel('Time [s]', fontsize=16)
fig.savefig('Time_Analysis')
