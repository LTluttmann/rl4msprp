import torch
import numpy as np
import copy
from rl4mixed.problems.dataset import BatchedInstances
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict
from collections import defaultdict
import contextlib


__helper__ = [
    "distance",
    "reward_fn",
    "route_idx_from_loc",
    "temporary_tour",
    "temporarily_destroy_node",
    "temporarily_destroy_task",
    "temporarily_destroy",
    "calc_decrease_dist",
    "calc_increase_dist",
]

__sets__ = [
    "Instance",
    "Node",
    "Item",
    "PickingTask",
    "Tour"
]

__all__ = __helper__ + __sets__


def distance(a,b):
    a = torch.Tensor(a) if not isinstance(a, torch.Tensor) else a
    b = torch.Tensor(b) if not isinstance(b, torch.Tensor) else b
    assert len(a.shape)==1
    assert len(b.shape)==1
    return (a - b).norm(p=2, dim=0).item()


def reward_fn(tour, agg=True):
    distances = torch.sqrt(torch.sum(torch.pow(tour[:-1]-tour[1:], 2), dim=1))
    if agg:
        return distances.sum(0)
    return distances

def route_idx_from_loc(route: torch.Tensor, loc: Tuple[float, float]):
    return route.numpy().tolist().index(loc)


@dataclass
class Node:
    idx: int
    loc: Tuple[float, float]
    supply: List[int]


@dataclass
class Item:
    idx: int
    demand: int

    @property
    def has_demand(self):
        return self.demand > 0


@dataclass
class Instance:
    nodes: List[Node]
    items: List[Item]
    kappa: int
    # supply: torch.Tensor

    @classmethod
    def initialize(cls, instance: BatchedInstances):
        assert not instance.is_batched
        assert not instance.is_flattened
        if instance.is_normalized:
            instance = instance.clone().unnormalize_batch().to("cpu")
        else:
            instance = instance.clone().to("cpu")


        items = [Item(i, demand.item()) for i,demand in enumerate(instance.demand)]
        nodes = [
            Node(i, loc.numpy().tolist(), supply.numpy().tolist()) 
            for i, (loc, supply) in enumerate(zip(instance.loc_w_depot, instance.supply_w_depot))
        ]

        kappa = copy.deepcopy(instance.capacity)

        return cls(nodes, items, kappa)
        

    @property
    def shelves(self) -> List[Node]:
        return self.nodes[1:]
    
    def get_node_from_id(self, id):
        return self.nodes[id]
        
    def get_item_from_id(self, id):
        return self.items[id]
    
    @property
    def depot(self):
        return self.nodes[0]
    
    @property
    def demand(self):
        return [item.demand for item in self.items]
    
    @property
    def items_w_demand(self):
        items_w_demand = [item for item in self.items if item.has_demand]
        np.random.shuffle(items_w_demand)
        return items_w_demand



@dataclass 
class PickingTask:
    node: Node
    item: Optional[Item] = None
    amount: Optional[int] = None
    _active: bool = False


    @property
    def active(self):
        return self._active

    def destroy(self):
        assert self.active, "cannot destroy an inactive task"
        self.item.demand += self.amount
        self.node.supply[self.item.idx] += self.amount
        self._active = False
        return self.amount

    def activate(self):
        assert not self.active, "cannot add an already active task"
        self.item.demand -= self.amount
        self.node.supply[self.item.idx] -= self.amount
        self._active = True
        

@dataclass
class Tour:
    idx: int
    capacity: int
    start_depot: Node
    schedule: List[PickingTask] = field(default_factory=lambda: list())
    dest_depot: Node = None

    def __post_init__(self):
        if self.start_depot is not None and self.dest_depot is None:
            self.dest_depot = self.start_depot

    def add_task(self, task: PickingTask, idx):
        if not task.active:
            task.activate()
        
        self.capacity -= task.amount
        self.schedule.insert(idx, task)


    def remove_task(self, task_or_idx: Union[int, PickingTask]) -> Tuple[PickingTask, int]:
        if isinstance(task_or_idx, PickingTask):
            assert task_or_idx in self.schedule, "given task to remove is not in schedule"
            task_idx = self.schedule.index(task_or_idx)
        elif isinstance(task, int):
            task_idx = task_or_idx
        else:
            raise ValueError("wrong data type passed")

        task = self.schedule.pop(task_idx)
        amount = task.destroy()
        self.capacity += amount
        return task, task_idx
    
    
    def remove_node(self, node: Union[int, Node]) -> List[Tuple[int, PickingTask]]:
        node_idx = node if isinstance(node, int) else node.idx
        node_tasks: List[PickingTask] = self.tasks_per_node[node_idx]
        old_node_positions: List[int] = []
        for task in node_tasks:
            _, pos = self.remove_task(task)
            old_node_positions.append(pos)
        return node_tasks, old_node_positions
    

    def add_node(self, node_tasks: List[Tuple[int, PickingTask]]):
        for idx, task in node_tasks:
            self.add_task(task, idx)


    def reset(self):
        for task in self.schedule:
            amount = task.destroy()
            self.capacity += amount
        self.schedule = list()


    def loc_from_idx(self, idx: int):
        return self[idx].node.loc


    def __len__(self):
        return len(self.schedule)
    
    def __getitem__(self, idx) -> PickingTask:
        return self.schedule[idx]

    
    @property
    def load(self):
        return sum([task.amount or 0 for task in self.schedule])
    
    @property
    def route(self):
        return self.get_route(unique_nodes=False, include_depot=True)
    
    def get_route(self, unique_nodes=False, include_depot=False, as_tensor=True) -> torch.Tensor:
        nodes = [task.node.loc for task in self.schedule]
        if unique_nodes:
            nodes = [nodes[i] for i in range(len(nodes)) if i==0 or nodes[i] != nodes[i-1]]
        if include_depot:
            nodes = [self.start_depot.loc] + nodes + [self.dest_depot.loc]
        if as_tensor:
            nodes = torch.Tensor(np.vstack(nodes))
        return nodes
    
    @property
    def distance(self):
        return reward_fn(self.route)
    
    @property
    def tasks_per_node(self) -> Dict[int, PickingTask]:
        nodes = defaultdict(list)
        for x in self.schedule:
            nodes[x.node.idx].append(x)
        return nodes
    
    @property
    def nodes(self) -> List[Node]:
        nodes = []
        for x in self.schedule:
            if x.node not in nodes:
                nodes.append(x.node)
        return nodes
    
    def get_nodes_from_locs(self, locs: List[Tuple[float, float]]):
        return [node for loc in locs for node in self.nodes if node.loc == loc]
    
    def reschedule_by_nodes(self, nodes: List[Node]):
        """ this function rearranges the schedule (the list of picking tasks) according
        to a route of nodes.
        :param List[Node] nodes: an list of nodes ordered according to the planned route
        """
        self.schedule = [task for node in nodes for task in self.tasks_per_node[node.idx]]


class TemporaryTour:
    def __init__(self, tour: Tour):
        self.tour = tour
        self.previous_state = None
        self.accept = False

    def __enter__(self):
        self.previous_state = copy.deepcopy(self.tour)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            # Exception occurred within the `with` block, no need to rollback changes
            return

        if not self.accept:
            # rollback
            self.tour.__dict__.clear()
            self.tour.__dict__.update(self.previous_state.__dict__)


@contextlib.contextmanager
def temporary_tour(tour: Tour):
    tour_copy = copy.deepcopy(tour)
    try: 
        yield tour_copy
    finally:
        if tour_copy.distance < tour.distance:
            tour.__dict__.clear()
            tour.__dict__.update(tour_copy.__dict__)


@contextlib.contextmanager
def temporarily_destroy_node(tour: Tour, node: Node):
    """this function temporarily destroys a node, leading to changes in the 
    respective supply and demand nodes as well as the tours capacity. When the 
    context calling this function closes, the node will be added to the same tour
    again, yielding a rollback to the state before calling this function. 
    """
    node_tasks, task_positions = tour.remove_node(node)
    try:
        yield tour
    finally:
        tour.add_node(list(zip(task_positions, node_tasks)))

@contextlib.contextmanager
def temporarily_destroy_task(tour: Tour, task: PickingTask):
    """this function temporarily destroys a task, leading to changes in the 
    respective supply and demand nodes as well as the tours capacity. When the 
    context calling this function closes, the task will be added to the same tour
    again, yielding a rollback to the state before calling this function. 
    """
    task, task_idx = tour.remove_task(task)
    try:
        yield tour
    finally:
        tour.add_task(task, task_idx)

def temporarily_destroy(tour, task_or_node):
    if isinstance(task_or_node, Node):
        return temporarily_destroy_node(tour, task_or_node)
    else:
        return temporarily_destroy_task(tour, task_or_node)


def calc_increase_dist(tour: Tour, loc: Tuple[float,float], idx: int, unique=False):
    """
    given a route, this function calculates for a given shelf and a given position at which the shelf is inserted
    in the route how much distance this shelf would add to the tour.
    :param route: current rout
    :param shelf: new node / shelf
    :param i: position to insert the new node at
    :param pack_station: pack station of the batch
    :return:
    """
    if len(tour) == 0:
        return distance(tour.start_depot.loc, loc) + distance(loc, tour.dest_depot.loc)
    
    tour_locs = tour.get_route(unique_nodes=unique, include_depot=False)
    # if the shelf at the current position of the tour is not equal to the candidate shelf, the distance added to
    # the tour by candidate shelf j at position i is calculated as follows: d_{i-1, j} + d{j, i} - d{i-1, i}
    # (pack stations are added manually as they are not "part" of route object)
    if idx == 0:
        add_dist = distance(tour.start_depot.loc, loc) + distance(loc, tour_locs[idx])
        subtr_dist = distance(tour.start_depot.loc, tour_locs[idx])
        
    elif idx == len(tour_locs):
        add_dist = distance(tour_locs[-1], loc) + distance(loc, tour.dest_depot.loc)
        subtr_dist = distance(tour_locs[-1], tour.dest_depot.loc)
    else:
        add_dist = distance(tour_locs[idx-1], loc) + distance(loc, tour_locs[idx])
        subtr_dist = distance(tour_locs[idx-1], tour_locs[idx])
    return add_dist - subtr_dist


def calc_decrease_dist(tour: Tour, node_task_or_idx: Union[Node, PickingTask, int], unique=False):
    """
    given a route, this function calculates for a given node idx how much the distance of the tour
    would decrease if this node would be removed.
    :param route: current rout
    :param shelf: new node / shelf
    :param i: position to insert the new node at
    :param pack_station: pack station of the batch
    :return:
    """

    if len(tour) == 0:
        return 0
    
    tour_locs = tour.get_route(unique_nodes=unique, include_depot=False)

    if isinstance(node_task_or_idx, PickingTask):
        idx = tour.schedule.index(node_task_or_idx)
    elif isinstance(node_task_or_idx, Node):
        assert unique
        idx = route_idx_from_loc(tour_locs, node_task_or_idx.loc)
    elif isinstance(node_task_or_idx, int):
        idx = node_task_or_idx
    else:
        raise ValueError("got unexpected input")
        
    loc = tour_locs[idx]
    # if the shelf at the current position of the tour is not equal to the candidate shelf, the distance added to
    # the tour by candidate shelf j at position i is calculated as follows: d_{i-1, j} + d{j, i} - d{i-1, i}
    # (pack stations are added manually as they are not "part" of route object)
    if idx == 0:
        if len(tour_locs) == 1:
            add_dist = distance(tour.start_depot.loc, tour.dest_depot.loc)
            subtr_dist = distance(tour.start_depot.loc, loc) + distance(loc, tour.dest_depot.loc)
        else:
            add_dist = distance(tour.start_depot.loc, tour_locs[idx+1])
            subtr_dist = distance(tour.start_depot.loc, loc) + distance(loc, tour_locs[idx+1])
        
    elif idx == len(tour_locs)-1:
        add_dist = distance(tour_locs[-2], tour.dest_depot.loc)
        subtr_dist = distance(tour_locs[-2], tour_locs[-1]) + distance(tour_locs[-1], tour.dest_depot.loc)
    else:
        add_dist = distance(tour_locs[idx-1], tour_locs[idx+1])
        subtr_dist = distance(tour_locs[idx-1], loc) + distance(loc, tour_locs[idx+1])
    return subtr_dist - add_dist

