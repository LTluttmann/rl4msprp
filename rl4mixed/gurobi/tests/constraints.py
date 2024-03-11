import re
import gurobipy as gp

from rl4mixed.settings import CAPACITY


def get_idx(var):
    return tuple(re.findall(r'\d+', var.VarName))

def test_capacity_constraint(model, vehicles):
    y_vals = gp.tupledict({get_idx(var): var.X for var in model.getVars() if "y" in var.VarName})
    for k in vehicles:
        vehicle_load = sum(y_vals.select('*', k))
        assert vehicle_load <= CAPACITY



