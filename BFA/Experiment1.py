import Candidate as Pop
import VisualizationGA as Viz
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt

# Parameters for the simulation (DESIGNER DEFINES THEM based on mission Analysis!)

NUM_AGENTS = 9                  # Initial Number of Interacting Agents
MAX_NUM_AGENTS = 10             # Initial Number of Interacting Agents
POPULATION_SIZE = 1             # Number of Topology Candidates
HIERARCHY1 = [3, 4, 5]        # Hierarchical Constraints
HIERARCHY2 = [6, 7, 8]        # Hierarchical Constraints
TEAM1 = [2, 3, 6]
GENERATION = 0
TEAMS = [TEAM1]
HIERARCHIES = [HIERARCHY1, HIERARCHY2]
CASE_1_H = [[6, 7, 8]]
CASE_2_H = [[3, 4, 5], [6, 7, 8]]
CASE_3_H = [[2, 3, 6], [3, 4, 5], [6, 7, 8]]

CASE_1_T = [[2, 6]]
CASE_2_T = [[2, 3, 6]]
CASE_3_T = [[4, 7], [5, 8]]

TIMES = []
AGENTS = []
COST = []
iteration = NUM_AGENTS

k = 1

file = open("Experiment_Data.csv", "a")
while iteration <= MAX_NUM_AGENTS:
    start = timer()
    COST_MAT = np.triu(np.ones((iteration, iteration)))  # Parameters for Cost
    print("Number of Agents: ", iteration)
    AGENTS.append(iteration)
    np.fill_diagonal(COST_MAT, 0)
    # Create a Population of Candidate Topologies for MAS-based On-board Software
    population = Pop.Candidates(iteration, POPULATION_SIZE, COST_MAT, CASE_2_T, CASE_2_H)
    population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
    end = timer()

    # Visualize the Candidate Topologies
    topologies = Viz.VisualizationGA(iteration)
    i = 0
    while i < POPULATION_SIZE:
        candidate_top = population.get_chromosomes()[i].get_genes()
        interact = population.get_chromosomes()[i].get_master_nodes()
        team_edges = population.get_chromosomes()[i].get_team_edges()
        h_edges = population.get_chromosomes()[i].get_hierarchy_edges()
        topologies.visualize_topology(candidate_top, interact, "Candidate_Arch_"+str(iteration)+"_"+str(k)+"_", i, team_edges, h_edges)
        file.write(str(k)+"; "+str(iteration)+"; "+str(end-start)+"; "+str(population.get_chromosomes()[0].get_fitness())+"; "+str(population.get_chromosomes()[0].get_total_cost())+"; "+str(population.get_chromosomes()[0].get_fixed_cost())+"\n")
        i += 1
    population.view_population(population, population.get_chromosomes(), GENERATION)
    # print("Budget for Total Communication Cost: ", population.get_chromosomes()[0].get_target_cost())
    print("\nElapsed Time", end-start, "seconds")
    TIMES.append(end-start)
    COST.append(population.get_chromosomes()[0].get_fitness())
    iteration += 1
print(TIMES)
print(AGENTS)
print(COST)
fig = plt.figure()
plt.xticks(AGENTS)
plt.plot(AGENTS, TIMES)
fig.suptitle('Time to Find a Solution with BF Search', fontsize=18)
plt.xlabel('Number of Agents', fontsize=14)
plt.ylabel('Time [s]', fontsize=16)
fig.savefig("Time_Analysis_"+str(k))
print(k)
file.close()
print("End of Experiment")
