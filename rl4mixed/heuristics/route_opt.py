import numpy as np
import torch
from typing import List
import itertools
from rl4mixed.heuristics.misc import Tour, reward_fn


class RouteOptimizer:


    # routing optimization # TODO different script
    def two_opt(self, tour: Tour):
        """
        implements the two opt heuristic. The algorithm iterates over every pair of edges of the tour and interchanges
        them.
        :param tour: the batch of the neighborhood solution
        :param currentSolution: the current solution to be improved
        :param T: current temperature
        """
        curr_fitness = tour.distance
        curr_tour = tour.get_route(include_depot=False, unique_nodes=True, as_tensor=False)
        if len(curr_tour) < 3:
            return
        for i in range(len(curr_tour) - 2):
            for j in range(i + 2, len(curr_tour)):
                cand_tour = tour.get_route(include_depot=False, unique_nodes=True, as_tensor=False)
                cand_tour[i] = cand_tour[j]
                reverse_order = [curr_tour[m] for m in reversed(range(i, j))]
                cand_tour[i + 1 : j + 1] = reverse_order
                cand_fitness = reward_fn(torch.Tensor([tour.start_depot.loc] + cand_tour + [tour.dest_depot.loc]))

                if cand_fitness < curr_fitness and not torch.isclose(cand_fitness, curr_fitness):
                    # apply new route
                    nodes_new_order = tour.get_nodes_from_locs(cand_tour)
                    tour.reschedule_by_nodes(nodes_new_order)
                    return True, True
                else:
                    assert cand_tour != tour.get_route(include_depot=False, unique_nodes=True, as_tensor=False)
        return False, False


    def swap(self, tour: Tour):
        curr_fitness = tour.distance
        curr_tour = tour.get_route(include_depot=False, unique_nodes=True, as_tensor=False)
        if len(curr_tour) < 2:
            return
        
        pairs = list(itertools.combinations(range(len(curr_tour)), 2))
        pairs = np.random.permutation(pairs)
        for idx1, idx2 in pairs:

            cand_tour = tour.get_route(include_depot=False, unique_nodes=True, as_tensor=False)
            cand_tour[idx1], cand_tour[idx2] = cand_tour[idx2], cand_tour[idx1]
            cand_fitness = reward_fn(torch.Tensor([tour.start_depot.loc] + cand_tour + [tour.dest_depot.loc]))

            if cand_fitness < curr_fitness and not torch.isclose(cand_fitness, curr_fitness):
                # apply new route
                nodes_new_order = tour.get_nodes_from_locs(cand_tour)
                tour.reschedule_by_nodes(nodes_new_order)
                return True, True
            else:
                assert cand_tour != tour.get_route(include_depot=False, unique_nodes=True, as_tensor=False)

        return False, False


    def relocate(self, tour: Tour):
        curr_fitness = tour.distance
        curr_tour = tour.get_route(include_depot=False, unique_nodes=True, as_tensor=False)

        for idx1 in range(len(curr_tour)):
            for idx2 in set(range(len(curr_tour))).difference([1]):

                if idx1 == idx2:
                    continue

                cand_tour = tour.get_route(include_depot=False, unique_nodes=True, as_tensor=False)

                cand_tour.insert(idx2, cand_tour.pop(idx1))

                cand_fitness = reward_fn(torch.Tensor([tour.start_depot.loc] + cand_tour + [tour.dest_depot.loc]))

                if cand_fitness < curr_fitness and not torch.isclose(cand_fitness, curr_fitness):
                    # apply new route
                    nodes_new_order = tour.get_nodes_from_locs(cand_tour)
                    tour.reschedule_by_nodes(nodes_new_order)
                    return True, True
                else:
                    assert cand_tour != tour.get_route(include_depot=False, unique_nodes=True, as_tensor=False)

        return False, False


    @staticmethod
    def sort_duplicates(lst):
        return sorted(lst, key=lambda x: (lst.index(x), lst.count(x) > 1))

    def cluster_nodes(self, tour: Tour):
        route = tour.get_route(unique_nodes=False, include_depot=False)
        route = route.numpy().tolist()
        sorted_idx_and_vals = sorted(
            enumerate(route), key=lambda x: (route.index(x[1]), route.count(x[1]) > 1))
        
        tour.schedule = [tour.schedule[i] for i, _ in sorted_idx_and_vals]


    def get_weight_update(self, accepted, improved, is_best):
        omega1, omega2, omega3, omega4 = 5, 3, 0, 0
        if is_best:
            return omega1
        elif improved:
            return omega2
        elif accepted:
            return omega3
        else:
            return omega4
        

    def route_optimization(
        self,
        solution: List[Tour],
        max_iterations=5,
        max_it_without_improvement=3,
        lambda_param=0.75,
        init_weights=1,
    ):


        operations = [self.two_opt, self.swap, self.relocate]

        weights = [init_weights] * len(operations)

        for tour in solution:

            best_fit = tour.distance
            curr_fit = tour.distance

            iteration = 0
            it_without_improvement = 0

            while (
                iteration < max_iterations
                and it_without_improvement < max_it_without_improvement
            ):
                is_best = False
                probs = [x / sum(weights) for x in weights]
                operation = np.random.choice(operations, p=probs)
                if len(tour.get_route(unique_nodes=True, include_depot=False)) > 2:

                    accepted, improved = operation(tour)

                    if accepted:
                        curr_fit = tour.distance
                        if curr_fit < best_fit:

                            is_best = True
                            best_fit = curr_fit

                    weight_update = self.get_weight_update(accepted, improved, best_fit==is_best)
                    weights[operations.index(operation)] = (
                        lambda_param * weights[operations.index(operation)]
                        + (1 - lambda_param) * weight_update
                    )

                    if not improved:
                        it_without_improvement += 1 
                    else:
                        it_without_improvement = 0

                    iteration += 1
                else:
                    break
