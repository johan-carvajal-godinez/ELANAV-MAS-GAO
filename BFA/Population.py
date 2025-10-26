import Chromosome as Chr
import networkx as nx
import numpy as np


class Population:   # Class For Creating a Feasible Population of MAS-based Topologies for the GAO
    def __init__(self, num_agents, pop_size, cost_par_mat, team_int_constraints, hierarchical_int_constraints):
        self._chromosomes = []
        i = 0
        trial = 0
        while i < pop_size:
            candidate_crom = Chr.Chromosome(num_agents, cost_par_mat)
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
            # Verify Constraints Conditions for Candidate Generation
            if candidate_crom.get_total_cost() <= candidate_crom.get_target_cost():
                if not candidate_crom.get_coherence():
                    trial += 1
                    print("Trial Number:", trial)
                else:
                    # print("Coherent")
                    self._chromosomes.append(candidate_crom)
                    # print(candidate_crom.get_fixed_cost(), candidate_crom.get_total_cost())
                    i += 1
            else:
                trial += 1
                print("Trial Number:", trial)

    def _test_chromosome(self):  # Test if an MAS-based Candidate Architecture is able to satisfy the Constraints
        self._valid_gene = True
        return self._valid_gene

    def get_chromosomes(self):  # Getter for a list of candidate MAS-based Architectures
        return self._chromosomes

    @staticmethod
    def view_population(self, chromos, gen_number):  # Print Method for the Population Class
        print("Interaction Constraints: ", chromos[0].get_constraints())
        print("Coordination Nodes: ", sorted(chromos[0].get_master_nodes()))
        print("\n-----------------------------------------------------------------------")
        print("Generation #", gen_number, "| Fittest chromosome Cost:", chromos[0].get_fitness())
        print("-----------------------------------------------------------------------")
        i = 0
        for x in chromos:
            print("Chromosome #", i, " :", x, "| Fitness:", x.get_fitness(), " | Degree Range:", x.get_degree_range())
            i += 1
