import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
from itertools import permutations
import gurobipy as gp
from gurobipy import GRB
import logging
import pandas as pd
from rl4mixed.settings import *    
from rl4mixed.problems.dataset import BatchedInstances
from rl4mixed.problems.vrp import InstanceSolution

log = logging.getLogger(__name__)


@dataclass
class SolverInput:
    skus: List[int]
    vehicles: List[int]
    shelves: List[int]
    nodes: List[int]
    arcs: List[tuple]
    shelves_of_item: Dict[int, list]
    demand_of_item: Dict[int, int]
    capacity: int
    supply_at_shelf: List[int]  
    distances: List[List[float]]
    item_at_node: Dict[int, int]

    def add_vehicle(self):
        self.vehicles.append(len(self.vehicles))


@dataclass
class SolverOutput:
    visit: gp.tupledict
    taken_units: gp.tupledict



@dataclass
class GurobiSolution:
    grb_output: SolverOutput
    grb_input: SolverInput
    grb_model: gp.Model



def solver_model(d: SolverInput) -> Tuple[gp.Model, Tuple[Any, Any]]:

    BigM = d.capacity

    m = gp.Model('MIXED')

    x = {}
    for i in d.nodes:
        for j in d.nodes:
            for k in d.vehicles:
                x[i,j,k] = m.addVar(vtype=GRB.BINARY, name=f'x[{i},{j},{k}]')

    # y = {}
    # for p in d.skus:
    #     for k in d.vehicles:
    #         y[p,k] = m.addVar(lb=0, vtype=GRB.SEMIINT, name=f'y[{p},{k}]')

    y = {}
    for s in d.shelves:
        for k in d.vehicles:
            y[s,k] = m.addVar(lb=0, vtype=GRB.SEMIINT, name=f'y[{s},{k}]')

    # objective funcntion
    m.setObjective(gp.quicksum(x[i,j,k]*d.distances[i,j] for i,j in d.arcs for k in d.vehicles))

    # CONSTRAINTS
    # network flow constraint
    m.addConstrs(gp.quicksum(x[i,j,k] for j in d.nodes if j != i) == gp.quicksum(x[j,i,k] for j in d.nodes if j != i) for i in d.nodes for k in d.vehicles)
    # max one visit of node and vehicle
    m.addConstrs(gp.quicksum(x[i,j,k] for j in d.nodes) <= 1 for i in d.nodes for k in d.vehicles)
    # dont take more units than are available at shelf
    m.addConstrs(gp.quicksum(y[i,k] for k in d.vehicles) <= d.supply_at_shelf[i] for i in d.shelves)
    m.addConstrs(y[j,k] <= BigM * gp.quicksum(x[i,j,k] for i in d.nodes) for j in d.shelves for k in d.vehicles)
    # m.addConstrs(gp.quicksum(y[p,k] for k in d.vehicles) <= gp.quicksum(d.supply_at_shelf[j]*x[i,j,k] for i in d.nodes for j in d.shelves_of_item[p] for k in d.vehicles) for p in d.skus)
    # meet the demand exactely
    m.addConstrs(gp.quicksum(y[i,k] for i in d.shelves_of_item[p] for k in d.vehicles) == d.demand_of_item[p] for p in d.skus)
    # dont exceed the capacity of the vehicles
    m.addConstrs(gp.quicksum(y[i,k] for i in d.shelves) <= d.capacity for k in d.vehicles)
    # dont allow for self-visits
    m.addConstrs(x[i,i,k]==0 for i in d.nodes for k in d.vehicles)
    # let each vehicle leave the depot once
    m.addConstrs(gp.quicksum(x[0,j,k] for j in d.shelves ) == 1 for k in d.vehicles)
    # avoid that a node is visited and no units are taken (might happen if distance to node is zero)
    m.addConstrs(x[i,j,k] <= y[j,k] for i in d.nodes for j in d.shelves for k in d.vehicles)

    m._x = x

    m.Params.LazyConstraints = 1

    return m, (x,y)                                                                                                                                                                                                                              


def transform_tensorized_data(data: BatchedInstances):

    assert not data.is_batched
    data = data.clone().to("cpu")
    if not data.is_flattened:
        data = data.flatten()
    if data.is_normalized:
        data = data.unnormalize_batch()
    
    loads = data.original_capacity

    P = data.demand.nonzero().squeeze(1).numpy()
    # exclude nodes of items with no demand
    shelve_idx = [x.item() in P for x in data.item_ids]
    n = data.supply.size(0)  # only number of supply nodes
    N = [i for i in range(1, n+1)]  # dont include depot here...
    # exclude storage positions of skus without demand
    N = np.array(N)[shelve_idx].tolist()
    V = [0] + N  # ..but here

    c = torch.cdist(data.loc_w_depot, data.loc_w_depot)

    A = [(i,j) for i in V for j in V]  # all edges
    # shelves containing item
    N_p = {p: data.supply_w_depot[:,p].nonzero().squeeze(1).numpy() for p in P}
    # supply of item / shelf combination
    q = data.supply_w_depot.sum(1).numpy()
    # demand of item
    r_p = {p: data.demand[p].item() for p in P}

    num_tours_lower_bound = np.ceil(sum(list(r_p.values())) / loads).astype("int")
    K = list(range(num_tours_lower_bound))

    item_at_node = {i[0].item():i[1].item() for i in data.supply_w_depot.nonzero()}

    transformed_data = SolverInput(
        skus=P,
        vehicles=K,
        shelves=N,
        nodes=V,
        arcs=A,
        shelves_of_item=N_p,
        demand_of_item=r_p,
        capacity=loads,
        supply_at_shelf=q,
        distances=c,
        item_at_node=item_at_node
    )

    return transformed_data


# Callback - use lazy constraints to eliminate sub-tours
def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        vals = model.cbGetSolution(model._x)
        edges = gp.tuplelist((i, j, k) for i, j, k in vals.keys() if vals[i, j, k] > 0.5)
        vehicles = list(set([k for i,j,k in model._x.keys()]))
        for k in vehicles:
            # edges_of_k = edges.select('*','*',k).select('*','*')
            edges_of_k = gp.tuplelist((i, j) for i, j, k_ in edges if k_ == k)
            tour, is_subtour = subtour(edges_of_k)
            # subtour elimination constraint
            if is_subtour:
                for _k_ in vehicles:
                    model.cbLazy(
                        gp.quicksum(
                            model._x[i, j, _k_] 
                            for i, j in permutations(tour, 2)
                        ) <= len(tour)-1
                    )           


# Given a tuplelist of edges, find the shortest subtour not containing depot (0)
def subtour(edges):
    if not isinstance(edges, gp.tuplelist): edges = gp.tuplelist(edges)
    unvisited = list(set([i for key in edges for i in key]))
    cycle = range(len(unvisited)+1)  # initial length has 1 more city
    is_subtour = False
    while unvisited:
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            #if current != 0:
            try:
                unvisited.remove(current)
            except:
                pass
            neighbors = [j for i, j in edges.select(current, '*')
                         if j == 0 or j in unvisited]
        if 0 not in thiscycle and len(cycle) > len(thiscycle):
            is_subtour = True
            cycle = thiscycle
    return thiscycle, is_subtour


def get_complete_tour_idx(solution: SolverOutput, d: SolverInput):
    tours_and_units = []
    
    x = solution.visit
    y = solution.taken_units

    assert hasattr(x[x.keys()[0]], "X"), log.info(
        "Something is wrong with the solution", 
        type(x), x[x.keys()[0]]
    )

    for k in d.vehicles:
        selected = gp.tuplelist((i,j) for i,j,k_ in x if x[i,j,k].X > 0.5 and k_==k)
        tour, is_subtour = subtour(selected)

        assert not is_subtour

        units = []
        for node in tour:
            if node == 0:
                units.append(0.0)
            else:
                item = d.item_at_node[node]
                unit = y.select(node,k)[0].X
                assert unit > 0, f"Zero units are taken at node {node}"
                # sometime gurobi spits out weird floats. Round here
                units.append(round(unit))
                
        tour_and_units = list(zip(tour, units))
        tours_and_units.append(tour_and_units)

    tours_and_units = [node for subtour in tours_and_units for node in subtour]
    tours_and_units = [tours_and_units[i] for i in range(len(tours_and_units))
                       if i == 0 or not (tours_and_units[i][0] == 0 and tours_and_units[i-1][0] == 0)]

    return tours_and_units


def solve(instance, timeout=None, verbose=True, mipfocus=False, add_to_timeout=0, retry_w_mipfocus=False) -> Tuple[InstanceSolution, bool]:
    if timeout == 0:
        return InstanceSolution(reward=None, tour_and_units=None, runtime=0), False

    d: SolverInput = transform_tensorized_data(instance)

    found_opt = False

    while True:
        m, vars = solver_model(d)

        if not verbose:
            m.Params.LogToConsole = 0

        x,y = vars
        if timeout is not None:
            m.setParam('TimeLimit', timeout)

        if mipfocus:
            m.setParam('MIPFocus', 1)

        m.optimize(subtourelim)
        

        # check if a solution was found, otherwise raise time limit, retry with mipfocus or return empty solution
        if not m.SolCount > 0:

            if retry_w_mipfocus:
                return solve(instance, timeout, mipfocus=True, add_to_timeout=0, retry_w_mipfocus=False)
            
            elif add_to_timeout > 0:
                log.info(f"No solution was found. Raise timelimit by {add_to_timeout} seconds")
                timeout += add_to_timeout
                continue

            else:
                # add empty solution
                return InstanceSolution(reward=None, tour_and_units=None, runtime=m.Runtime), found_opt
            
        elif m.MIPGap > 1e-5: # use small threshold to avoid weird float behavior
            log.info("Preempting...Found non-optimal solution with obj. value %s and gap of %s percent" % 
                    (round(m.ObjVal,2), round(m.MIPGap*100,2)))
            break

        else:
            found_opt = True
            log.info("Found optimal solution with obj. value: %s" % m.ObjVal)
            break
    
    m.update()
    x=gp.tupledict(x)
    y=gp.tupledict(y)
    sol = SolverOutput(x,y)
    # infer total tour information from sovler solution
    tour_and_units = get_complete_tour_idx(sol, d)
    tests(x, y, tour_and_units, d, instance)
    # return standardized solution output
    return InstanceSolution(reward=m.ObjVal, tour_and_units=tour_and_units, runtime=m.Runtime), found_opt


def tests(x, y, tour_and_units, d:SolverInput, instance: BatchedInstances):

    # test whether no subtours exists
    for k in d.vehicles:
        selected = gp.tuplelist((i,j) for i,j,k_ in x if x[i,j,k].X > 0.5 and k_==k)
        _, is_subtour = subtour(selected)

        assert not is_subtour

    # test whether capacity limit is not exceeded
    for k in d.vehicles:

        load = sum([round(i.X) for i in y.select('*',k)])
        log.info(f"utilization of vehicle {k} is {load/d.capacity*100}%")
        assert load <= d.capacity, k

    # test whether demand is exactly met
    for p in d.skus:
        units_at_shelves = [y.select(i,'*') for i in d.shelves_of_item[p]]
        assert sum([round(i.X) for l in units_at_shelves for i in l]) == d.demand_of_item[p], p

    # test whether supply of shelves is not exceeded
    taken_units = pd.DataFrame(tour_and_units).groupby(0).agg("sum")
    supply = pd.DataFrame(zip(list(range(len(d.supply_at_shelf))), d.supply_at_shelf))
    jj = taken_units.reset_index().merge(supply, on=0, how='left')
    assert all(jj.iloc[:,1] <= jj.iloc[:,2])

    # test that we did not mess up the indices of skus, shelves, storage positions...
    tours = [x[0] for x in tour_and_units if x[0] != 0]
    picked_skus = instance.item_ids.gather(0, torch.Tensor(tours).to(torch.int64)-1)
    assert all([sku.item() in d.skus for sku in picked_skus])