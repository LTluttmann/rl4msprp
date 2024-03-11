import numpy as np
from copy import deepcopy
import random
import logging
from typing import Union
import time
import itertools
from rl4mixed.heuristics.greedy import ConstructionHeuristic, shuffle_dict
from rl4mixed.heuristics.misc import *
from rl4mixed.heuristics.route_opt import RouteOptimizer
from rl4mixed.problems.vrp import InstanceSolution

log = logging.getLogger(__name__)


class ImprovementHeuristic(ConstructionHeuristic):
    def __init__(self, batch_instance, seed=1234567) -> None:
        super().__init__(batch_instance, seed)
        super().construct_solution()
        self.route_optimizer = RouteOptimizer()
        self.min_batches = np.ceil(sum(self.instance_copy.demand)/self.instance_copy.kappa).astype("int")
        self.operators = [
            self.swap_shelves,
            self.swap_nodes,
            self.swap_task
        ]

        self.weights = [1] * len(self.operators)

    def check_swap_feasibility(self, 
                               tour1: Tour, 
                               tour2: Tour, 
                               task_or_node1: Union[Node, PickingTask], 
                               task_or_node2: Union[Node, PickingTask]):
        if isinstance(task_or_node1, Node):
            weight1 = sum([x.amount for x in tour1.tasks_per_node[task_or_node1.idx]])
            weight2 = sum([x.amount for x in tour2.tasks_per_node[task_or_node2.idx]])
        else:
            weight1 = task_or_node1.amount
            weight2 = task_or_node2.amount

        # check weight constraints
        weight_diff = weight1 - weight2

        if weight_diff <= 0: # s2 heavier, check if fits in 1
            feasible = abs(weight_diff) <= tour1.capacity
        else: # s1 heavier, check if fits in 2
            feasible = weight_diff <= tour2.capacity

        return feasible
    

    def calc_savings_of_swap(self,
                             tour1: Tour,
                             tour2: Tour,
                             task_or_node1: Union[Node, PickingTask],
                             task_or_node2: Union[Node, PickingTask]):

        feasible = self.check_swap_feasibility(tour1, tour2, task_or_node1, task_or_node2)
        
        if feasible:

            s1_savings = calc_decrease_dist(tour1, task_or_node1, unique=isinstance(task_or_node1, Node))
            s2_savings = calc_decrease_dist(tour2, task_or_node2, unique=isinstance(task_or_node2, Node))

            with temporarily_destroy(tour1, task_or_node1) as tour1:
                with temporarily_destroy(tour2, task_or_node2) as tour2:     
                    if isinstance(task_or_node1, Node):
                        # NOTE: the idea is to switch nodes, since all items picked up there dont add additional distance 
                        # to the tour. Therefore, we assume that all items belonging to the node (tasks of that node) are 
                        # adopted by the new tour. Therefore, we dont calc min_possible_dist here!
                        *return_vals2, cost1 = self.get_min_dist_position(tour1, task_or_node2.loc)
                        *return_vals1, cost2 = self.get_min_dist_position(tour2, task_or_node1.loc)
                    elif isinstance(task_or_node1, PickingTask):
                        # TODO need to update after cheapest insertion, otherwise it might happen that two tasks
                        # NOTE: might not be necessary when swaps of same skus are not considered
                        *return_vals2, cost1 = self.cheapest_insertion(task_or_node2.item, tour1, min_supply=task_or_node2.amount)
                        *return_vals1, cost2 = self.cheapest_insertion(task_or_node1.item, tour2, min_supply=task_or_node1.amount)

            total_savings = (s1_savings + s2_savings) - (cost1 + cost2)
        else:
            return_vals1 = [None]
            return_vals2 = [None]
            total_savings = -float("inf")

        return (total_savings, *return_vals1, *return_vals2)
    

    def switch_task(self, alt_tour: Tour=None, epsilon=1e-6):

        found_better_incumbent = False

        if len(self.solution) < 2:
            return found_better_incumbent
        
        tasks_feasible_to_swap: List[Tuple[PickingTask, Tour, List[Tour]]] = []
        for idx, tour in enumerate(self.solution):
            if alt_tour is not None and alt_tour.idx == tour.idx:
                continue
            
            for task in tour.schedule:

                if alt_tour is not None:
                    pot_tours = [alt_tour] if task.amount <= alt_tour.capacity else []
                else:
                    other_tours = [x for alt_idx,x in enumerate(self.solution) if alt_idx!=idx]
                    pot_tours = [alt_tour for alt_tour in other_tours if task.amount <= alt_tour.capacity]

                if len(pot_tours) > 0:
                    tasks_feasible_to_swap.append((task,tour, pot_tours))

        task_swap_cache = []
        for task, tour, pot_tours in tasks_feasible_to_swap:
            savings = calc_decrease_dist(tour, task) 

            with temporarily_destroy_task(tour, task):

                other_tours = [x for alt_idx, x in enumerate(self.solution) if alt_idx!=idx]

                total_savings = {}
                for idx, alt_tour in enumerate(pot_tours):
                    # TODO the meet_demand constraint might lead to bad results. Clever trick is needed here!
                    min_shelf, min_pos, min_dist = self.cheapest_insertion(task.item, alt_tour, min_supply=task.amount)
                    total_savings[idx] = (min_shelf, min_pos, savings-min_dist)
                
                total_savings = shuffle_dict(total_savings)
                new_tour_idx, cache = max(total_savings.items(), key=lambda x: x[1][2])
                min_shelf, min_pos, total_savings = cache

            if self.accept(total_savings + epsilon):
                new_tour = pot_tours[new_tour_idx]        
                task_swap_cache.append((task, tour, new_tour, min_shelf, min_pos, total_savings))

        if len(task_swap_cache) > 0:
            np.random.shuffle(task_swap_cache)
            task, tour, new_tour, min_shelf, min_pos, savings = max(task_swap_cache, key=lambda x: x[-1])
            tour.remove_task(task)
            self.update_step(task.item, new_tour, min_shelf, min_pos)
            found_better_incumbent = True
            
        return found_better_incumbent
        

    def swap_shelves(self, T=None):

        for tour in self.solution:

            for node in tour.nodes:

                tasks, _ = tour.remove_node(node)
                items = [task.item for task in tasks]
            
                for item in items:
                    while item.demand > 0:
                        self.greedy_insert_item_into_tour(item, tour)



    @staticmethod
    def accept(savings, T=None):
        if savings > 0:
            return True
        if T is None:

            if savings == 0:
                return bool(np.random.randint(0,2))     
                       
            else:
                return False
        else:
            return np.random.random() <= np.exp(-abs(savings) / T)
    


    def swap_task(self, T=None):
        if len(self.solution) < 2:
            return
        
        # sample two tours to swap tasks between
        tours_to_swap: List[Tour] = random.sample(self.solution, 2)
        
        candidate_swap_tasks = list(itertools.product(*[tour.schedule for tour in tours_to_swap]))
        possible_swap_tasks: List[Tuple[float, PickingTask, PickingTask, Tuple]] = []
        for t1, t2 in candidate_swap_tasks:
            # doesnt really make sense to swap the same sku. Rather, we need a heuristic checking if the
            # shelf assignment is optimal. Same sku swap also leads to errors
            if t1.item.idx == t2.item.idx: continue
            tour1 = tours_to_swap[0]
            tour2 = tours_to_swap[1]

            savings, *return_vals = self.calc_savings_of_swap(tour1, tour2, t1, t2)

            if self.accept(savings, T):
                possible_swap_tasks.append( (savings, t1, t2, *return_vals) )

        if len(possible_swap_tasks) > 0:
            np.random.shuffle(possible_swap_tasks)
            savings, t1, t2, *return_vals = max(possible_swap_tasks, key=lambda x: x[0])
            t1_shelf, t1_pos, t2_shelf, t2_pos = return_vals

            tour1.remove_task(t1)
            tour2.remove_task(t2)

            self.update_step(t1.item, tour2, t1_shelf, t1_pos, update_from_swap=True)
            self.update_step(t2.item, tour1, t2_shelf, t2_pos, update_from_swap=True)
            return savings
        


    def swap_nodes(self, T=None):
        if len(self.solution) < 2:
            return
        # sample two tours to swap nodes between
        tours_to_swap: List[Tour] = random.sample(self.solution, 2)

        candidate_swap_nodes = list(itertools.product(*[tour.nodes for tour in tours_to_swap]))
        possible_swap_nodes = []
        for n1, n2 in candidate_swap_nodes:
            tour1 = tours_to_swap[0]
            tour2 = tours_to_swap[1]

            total_savings, n1_new_pos, n2_new_pos = self.calc_savings_of_swap(tour1, tour2, n1, n2)

            if self.accept(total_savings, T):
                possible_swap_nodes.append( (total_savings, n1, n2, n1_new_pos, n2_new_pos) )

        if len(possible_swap_nodes) > 0:
            np.random.shuffle(possible_swap_nodes)
            total_savings, n1, n2, n1_new_pos, n2_new_pos = max(possible_swap_nodes, key=lambda x: x[0])

            node1_tasks, _ = tour1.remove_node(n1)
            node2_tasks, _ = tour2.remove_node(n2)

            for i, task in enumerate(node1_tasks):
                self.update_step(task.item, tour2, n1, n1_new_pos+i, to_satisfy=task.amount)
            for i, task in enumerate(node2_tasks):
                self.update_step(task.item, tour1, n2, n2_new_pos+i, to_satisfy=task.amount)
            return total_savings


    def destroy_and_repair(self):
        instance_copy = deepcopy(self)
        solution = sorted(instance_copy.solution, key=lambda x: len(x))
        improved = False
        for tour in solution:
            items = [task.item for task in tour.schedule]
            tour.reset()
            np.random.shuffle(items)
            for item in items:
                while item.demand > 0:
                    instance_copy.greedy_insert_item_into_tour(item, tour)

            found_better_incumbent = True
            while found_better_incumbent:
                found_better_incumbent = instance_copy.switch_task(alt_tour=tour, epsilon=0)

        instance_copy.randomized_local_search()
        self.route_optimizer.route_optimization(self.solution)

        if reward_fn(instance_copy.full_route) < reward_fn(self.full_route):
            self.__dict__ = deepcopy(instance_copy.__dict__)
            improved = True

        return improved



    def solve(self, max_iters=150, max_iters_wo_imp=20, max_runtime=500):
        assert len(self.solution) == self.min_batches
        random.seed(1234567)
        np.random.seed(1234567)

        start_time = time.time()

        iters = 0
        iters_without_imp = 0

        while (
            iters < max_iters
            and iters_without_imp < max_iters_wo_imp
            and (time.time() - start_time) < max_runtime
        ):

            improved = self.destroy_and_repair()

            if improved:
                iters_without_imp = 0
            else:
                iters_without_imp += 1

            iters += 1

        self.tests()
        reward = reward_fn(self.full_route).item()

        improvement_runtime = time.time() - start_time
        self.runtime += improvement_runtime

        log.info("Heuristic reward: %s after %s ierations" % (reward, iters))

        return InstanceSolution(reward=reward, tour_and_units=self.tours_and_units_flat, runtime=self.runtime)
    


    def randomized_local_search(self, max_iters=14, lamb=0.7, alpha=0.8, max_it_without_improvement=5):

        iters = 0
        it_without_improvement = 0
        curr_fit = reward_fn(self.full_route).item()

        T = 0.02 * curr_fit
        while iters < max_iters and it_without_improvement < max_it_without_improvement:
            probs = np.array(self.weights)/sum(self.weights)
            operator_idx = np.random.choice(np.arange(0, len(self.operators)), p=probs)
            operator = self.operators[operator_idx]

            savings = operator(T)
            if savings is not None:
                if savings > 0:
                    # it_without_improvement = 0
                    self.weights[operator_idx] = lamb * self.weights[operator_idx] + (1-lamb) * 3
                else:
                    self.weights[operator_idx] = lamb * self.weights[operator_idx] + (1-lamb) * 1
                    it_without_improvement += 1
            else:
                it_without_improvement += 1
                
            iters += 1
            T *= alpha