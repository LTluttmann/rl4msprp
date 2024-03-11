import torch
import numpy as np
from operator import itemgetter
import time
import copy
from typing import List, Tuple, Union
from rl4mixed.heuristics.misc import *
from rl4mixed.problems.vrp import InstanceSolution
from rl4mixed.problems.dataset import get_flat_ids


def shuffle_dict(d: dict):
    l = list(d.items())
    np.random.shuffle(l)
    d_shuffled = dict(l)
    return d_shuffled


class ConstructionHeuristic:
    def __init__(self, batch_instance, seed=1234567) -> None:
        self.orig_instance = batch_instance
        self.instance: Instance = Instance.initialize(batch_instance)
        self.instance_copy = copy.deepcopy(self.instance)
        self.solution: List[Tour] = []
        self.total_distance: float = 0
        self.runtime: float = 0

    @property
    def tours_and_units(self):
        tours_and_units = [(0, None, 0)]
        for tour in self.solution:
            tours_and_units += [(x.node.idx, x.item.idx, x.amount) for x in tour.schedule]
            tours_and_units += [(0, None, 0)]
        return tours_and_units
    
    @property
    def tours_and_units_flat(self):
        def _get_flat_idx(tour_and_units):
            tour_and_units_mapped = []
            for shelf, sku, amount in tour_and_units:
                node_id = get_flat_ids(shelf, sku, self.orig_instance)
                tour_and_units_mapped.append((node_id,amount))
            return tour_and_units_mapped
        
        return _get_flat_idx(self.tours_and_units)

    @property
    def full_route(self):
        start = [self.solution[0].route]
        others = [x.route[1:] for x in self.solution[1:]]
        full = torch.cat(start+others, 0)
        return full
    
    def get_min_dist_position(self, tour: Tour, loc: Tuple[float,float], ignore_node=[]):
        schedule_wo_ignore = [x for x in tour if x.node.idx not in ignore_node]
        tour_copy = copy.deepcopy(tour)
        tour_copy.schedule = schedule_wo_ignore
        added_dists = {
            j: calc_increase_dist(
                tour_copy, loc, j
            )
            for j in range(len(tour_copy) + 1 )
        }
        # some randomization in case multiple positions have same minimal distance
        added_dists = shuffle_dict(added_dists)
        min_pos, min_pos_dist = min(added_dists.items(), key=itemgetter(1))

        return min_pos, min_pos_dist
    

    def cheapest_insertion(self, item: Item, curr_tour: Tour, min_supply=None, ignore_node: Union[Node, List[Node]] = []) -> Tuple[Node, int, float]:

        ignore_node = [x.idx for x in ignore_node] if isinstance(ignore_node, List) else [ignore_node.idx]
        if min_supply is not None:
            shelves_w_item = [node for node in self.instance.shelves if node.supply[item.idx] >= min_supply and node.idx not in ignore_node]
        else:
            shelves_w_item = [node for node in self.instance.shelves if node.supply[item.idx] > 0 and node.idx not in ignore_node]

        assert len(shelves_w_item) > 0, "instance cannot be solved"

        shelf_add_distances = {}
        for shelf in shelves_w_item:
            min_pos, min_pos_dist = self.get_min_dist_position(curr_tour, shelf.loc, ignore_node=ignore_node)
            shelf_add_distances[shelf.idx, min_pos] = min_pos_dist

        # randomization if multiple keys with same minimal value
        shelf_add_distances = shuffle_dict(shelf_add_distances)
        min_shelf_idx, min_pos = min(shelf_add_distances, key=shelf_add_distances.get)
        min_dist = shelf_add_distances[min_shelf_idx, min_pos]
        min_shelf = self.instance.get_node_from_id(min_shelf_idx)
        return min_shelf, min_pos, min_dist


    def greedy_insert_item_into_tour(self, item: Item, curr_tour: Tour):
        min_shelf, min_pos, _ = self.cheapest_insertion(item, curr_tour)
        self.update_step(item, curr_tour, min_shelf, min_pos)


    def update_step(self, item: Item, curr_tour: Tour, min_shelf: Node, min_pos: int, update_from_swap=False, to_satisfy=None):

        demand_satisfied = min(item.demand, min_shelf.supply[item.idx])
        demand_satisfied = min(demand_satisfied, curr_tour.capacity)
        if to_satisfy is not None:
            assert demand_satisfied >= to_satisfy
            demand_satisfied = to_satisfy
        assert demand_satisfied > 0
        # update 
        task = PickingTask(min_shelf, item, demand_satisfied)
        curr_tour.add_task(task, min_pos)


    def tests(self):
        assert all([tour.load <= self.instance.kappa for tour in self.solution]), "too much load"
        assert all([a for tour in self.solution for a in [x.active for x in tour.schedule]]), "too much load"
        assert not any([item.has_demand for item in self.instance.items])


    def construct_solution(self):

        start_time = time.time()

        curr_tour_idx = 0

        while any([item.has_demand for item in self.instance.items]):

            curr_tour = Tour(curr_tour_idx, self.instance.kappa, self.instance.depot)

            init_item = np.random.choice(self.instance.items_w_demand)
            self.greedy_insert_item_into_tour(init_item, curr_tour)
            
            while curr_tour.capacity > 0 and any([item.has_demand for item in self.instance.items]):

                
                add_distance_per_item = dict()

                for item in self.instance.items_w_demand:

                    min_shelf, min_pos, dist = self.cheapest_insertion(item, curr_tour)
                    add_distance_per_item[item.idx, min_shelf.idx, min_pos] = dist

                best_item_idx, best_shelf_idx, best_pos = min(add_distance_per_item, key=add_distance_per_item.get)

                best_item = self.instance.get_item_from_id(best_item_idx)
                best_shelf = self.instance.get_node_from_id(best_shelf_idx)
                
                self.update_step(best_item, curr_tour, best_shelf, best_pos)
                #assert np.isclose(curr_tour.reward, curr_tour.distance)

            curr_tour_idx += 1
            self.solution.append(curr_tour)
            self.total_distance += curr_tour.distance
        
        construction_runtime = time.time() - start_time
        self.runtime += construction_runtime
        # stack tours
        reward = reward_fn(self.full_route)
        self.tests()

        return reward.item()
    
    def solve(self):
        reward = self.construct_solution()
        return InstanceSolution(reward=reward, tour_and_units=self.tours_and_units_flat, runtime=self.runtime)

