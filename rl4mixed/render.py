import torch
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.patches as patches
from typing import List
from rl4mixed.problems.vrp import InstanceSolution, InstanceSolutionSignature
from rl4mixed.problems.dataset import BatchedInstances
from PIL import Image

font = {"size": 12}

markers = [
    "o", "8", "s", "p", "h", "H", "D", 
    # "d", "v", "^", "<", ">", 
]

linestyles = [
    "solid",
    "dotted",
    "dashed",
    "dashdot"
]

MARKER_COLOR_MAP = pl.cm.Set2.colors
MARKER_FACE_COLOR = pl.cm.Set2.colors[2]
MARKER_LINE_COLOR = "black" # pl.cm.Set2.colors[1]
MARKER_TEXT_COLOR = "black"
ARROW_COLOR_MAP = pl.cm.Accent.colors
RECTANGLE_COLOR = "black"
DEPOT_MARKER_SIZE= 25
MARKER_SIZE = 20
    

def offset_fn(x, factor): return np.array(((-2)*(x % 2)+1)*np.ceil(x/2))*factor


def find_line(x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    b = y1-m*x1
    def line_fn(x): return m*x+b
    return line_fn


def get_full_tour_information(tour_indices: torch.Tensor, 
                              units: torch.Tensor) -> List[InstanceSolutionSignature]:
    
    tours_and_units = []
    
    for i in range(tour_indices.size(0)):

        instance_units = [round(i.item(), 2) for i in units[i].max(1).values]

        route = [int(i) for i in tour_indices[i]]
        route = list(zip(route, instance_units))

        route.insert(0, (0, 0.0))
        route.append((0, 0.0))

        tours_and_units.append(route)

    return tours_and_units


def prep_input(instance: BatchedInstances, solution: InstanceSolution):
    instance = instance.to("cpu")

    if instance.is_batched:
        assert len(instance) == 1
        instance = instance[0]

    if not instance.is_flattened:
        instance = instance.clone().flatten()
    else:
        instance = instance.clone()

    if instance.is_normalized:
        instance = instance.unnormalize_batch()

    tours_and_units = solution.tour_and_units
    assert tours_and_units is not None
    assert tours_and_units[-1][0] == 0
    assert tours_and_units[0][0] == 0

    loc = instance.loc
    items = instance.item_ids
    supplies = instance.supply.gather(1, items[:,None])
    demands = instance.demand.gather(0, items)
    
    df = pd.DataFrame(loc, columns=["x", "y"])
    df["item"] = items
    df["supply"] = supplies
    df["demand"] = demands
    # add depot
    depot = pd.DataFrame(np.empty((1, df.shape[1])), columns=df.columns)
    depot.loc[:,["x", "y"]] = instance.depot.numpy()
    df = pd.concat((depot, df))
    df.reset_index(inplace=True, drop=True)

    return df, tours_and_units



def calculate_control_point(a, b):
    """Function to calculate the control point dynamically based on distance (higher curvature
    for less distant points) and as being orthogonal to midpoint of a and b"""
    a = np.array(a)
    b = np.array(b)
    # Step 1: Calculate the midpoint of the line segment
    midpoint = (a + b) / 2

    # Step 2: Calculate the direction vector of the line segment
    point_distance = np.linalg.norm(b - a)
    direction = (b - a) / point_distance

    # Step 3: Rotate the direction vector by 90 degrees
    orthogonal_direction = np.array([-direction[1], direction[0]])

    # Step 4: Define the desired distance from the midpoint
    distance_from_midpoint = np.log10(point_distance) * 0.05
    switch = bool(np.random.randint(0,2))
    distance_from_midpoint = distance_from_midpoint * (-1) if switch else distance_from_midpoint
    # Calculate the final point
    orthogonal_point = midpoint + distance_from_midpoint * orthogonal_direction

    return orthogonal_point

# Function to calculate the number of points based on distance
def calculate_num_points(start, end):
    distance = np.linalg.norm(np.array(end) - np.array(start)) * 100
    min_points = 10  # Minimum number of points
    max_points = 100  # Maximum number of points
    return min(max_points, min_points + int(distance))

def get_num_tours(tours_and_units):
    tours = [i[0] for i in tours_and_units]
    num_tours = 0
    for i in range(len(tours)-1):
        if tours[i+1] == 0 and tours[i] != 0:
            num_tours +=1
    return num_tours


def render_solution(instance: BatchedInstances, 
                    solution: InstanceSolution, 
                    complex_legend=True,
                    annotate_nodes=True):

    df, tours_and_units = prep_input(instance, solution)

    groups = df.groupby(["x", "y"])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

    for _, group in groups:
        grouplen = len(group)
        group_idx = group.index[0]

        if group_idx == 0:
            df.loc[group_idx, "vis_x"] = group.x.values[0]
            df.loc[group_idx, "vis_y"] = group.y.values[0]           
            ax.plot(group.x, group.y, marker="*",
                    markerfacecolor=MARKER_FACE_COLOR,
                    linestyle="", label="depot",
                    ms=DEPOT_MARKER_SIZE,
                    color=MARKER_LINE_COLOR)
            
        else:

            num_rows = np.ceil(grouplen/2)
            num_cols = np.ceil(grouplen/num_rows)

            x = group.x.to_list()[0]
            y = group.y.to_list()[0]

            patch = patches.Rectangle((x-0.02, y-0.02),
                                      0.04 * num_cols,
                                      0.04 * num_rows,
                                      linewidth=1, 
                                      edgecolor=RECTANGLE_COLOR,
                                      facecolor='none',
                                      label="Shelf")

            ax.add_patch(patch)

            curr_row = 0
            curr_col = 0
            for node in group.iterrows():

                node_idx, node_obj = node

                node_x = node_obj.x + (0.04 * curr_col)
                node_y = node_obj.y + (0.04 * curr_row)

                if curr_col+1 >= num_cols:
                    curr_row += 1
                    curr_col = 0
                else:
                    curr_col += 1

                df.loc[node_idx, "vis_x"] = node_x
                df.loc[node_idx, "vis_y"] = node_y

                item = int(node_obj["item"])
                demand = int(node_obj["demand"])
                supply = int(node_obj["supply"])

                # plot the item in the shelf
                if complex_legend:
                    label = f"{item} ({demand})"
                else:
                    if annotate_nodes:
                        label = "SKU (supply)"
                    else:
                        label = "SKU"

                ax.plot(node_x,
                        node_y,
                        marker=markers[item%len(markers)], 
                        linestyle='-', 
                        markerfacecolor=MARKER_COLOR_MAP[item%len(MARKER_COLOR_MAP)],
                        ms=MARKER_SIZE, 
                        label=label,
                        color=MARKER_LINE_COLOR, 
                        lw=2)

                if annotate_nodes:
                    ax.annotate(str(supply),  
                                xy=(node_x, node_y),
                                color=MARKER_TEXT_COLOR,
                                # fontsize="small", 
                                horizontalalignment='center', 
                                verticalalignment='center')
                    
    if not complex_legend:
                    
        from collections import OrderedDict
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), labelspacing = 1.5, columnspacing=1.5, fontsize=14)

    # plot arrows to visualize network flow
    n = len(tours_and_units)

    hist = []
    arrows = {}
    tour_idx = -1
    for i in range(0, n-1):
        start = tours_and_units[i][0]
        dest = tours_and_units[i+1][0]
        if start == dest:
            continue
        elif start == 0:
            tour_idx += 1

        unit = abs(int(tours_and_units[i+1][-1]))

        start_point = df.loc[start, ["vis_x", "vis_y"]].values
        end_point = df.loc[dest, ["vis_x", "vis_y"]].values
        control_point = calculate_control_point(start_point, end_point)

        num_points = calculate_num_points(start_point, end_point)
        t = np.linspace(0, 1, num_points)
        x = (1 - t)**2 * start_point[0] + 2 * (1 - t) * t * control_point[0] + t**2 * end_point[0]
        y = (1 - t)**2 * start_point[1] + 2 * (1 - t) * t * control_point[1] + t**2 * end_point[1]

        # Calculate arrowhead position slightly before the end of the curve
        end_index = num_points - 3
        arrow_x = x[end_index]
        arrow_y = y[end_index]

        # Plot the curve up to the arrowhead position
        plt.plot(x[:end_index+1], 
                 y[:end_index+1], 
                 color=ARROW_COLOR_MAP[tour_idx])

        # Add an arrowhead slightly before the end of the curve
        arr = plt.arrow(x[end_index - 1],
                        y[end_index - 1],
                        arrow_x - x[end_index - 1],
                        arrow_y - y[end_index - 1],
                        head_width=0.01,
                        head_length=0.01,
                        fc=ARROW_COLOR_MAP[tour_idx],
                        ec=ARROW_COLOR_MAP[tour_idx],
                        alpha=0.95)

        arrows[f"{start} $\\rightarrow$ {dest} ({unit})"] = arr

        hist.append((start, dest))
        hist.append((dest, start))

    if complex_legend:

        legend1 = plt.legend(list(arrows.values()), list(arrows.keys()),
                            loc='center left', bbox_to_anchor=(1, 0.5),
                            title="Flow (Units)",fancybox=True, shadow=True)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        legend2 = ax.legend(by_label.values(), by_label.keys(),
                            title="Item (demand)", loc='upper center',
                            bbox_to_anchor=(0.5, -0.05),
                            fancybox=True, shadow=True, ncol=5)

        #ax.legend(title="Flow", loc='upper right')
        plt.gca().add_artist(legend1)
        bbox_extra_artists=(legend1,legend2)

    else:
        bbox_extra_artists=None
    
    ax.set_title(
        f"Reward: {round(solution.reward if not isinstance(solution.reward, torch.Tensor) else solution.reward.item(), 2) or 'unknown'}",
        fontsize=20
    )

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.05, 1.05)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, bbox_extra_artists=bbox_extra_artists, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)
