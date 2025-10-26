import Chromosome as chromosx


class Candidates:    # Class For Creating a Feasible Population of MAS-based Topologies for the GAO
    def __init__(self, num_agents, pop_size, cost_par_mat, team_int_constraints, hierarchical_int_constraints):
        self._chromosomes = []
        i = 0
        trial = 0
        while i < pop_size:
            candidate_crom = chromosx.Chromosome(num_agents, cost_par_mat)
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
            if candidate_crom.test_degree_coord():
                if candidate_crom.test_degree_funct():
                    self._chromosomes.append(candidate_crom)
                    i += 1
                else:
                    trial += 1
                    print(trial)
            else:
                trial += 1
                print(trial)

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
            print("Chromosome #", i, " :", x, "| Fitness:", x.get_fitness())
            i += 1
