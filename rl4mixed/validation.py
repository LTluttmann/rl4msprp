import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import hydra
from omegaconf import DictConfig
from collections import  defaultdict
from tqdm import tqdm
import neptune
import logging
import time
from tqdm import tqdm
from typing import List
from rl4mixed.render import render_solution, get_full_tour_information
from rl4mixed.problems.dataset import (
    MixedShelvesDataset, 
    BatchedInstances, 
    get_flat_ids
)
from rl4mixed.problems.vrp import calc_reward, InstanceSolution
import rl4mixed.settings as opt
from rl4mixed.utils import set_decode_type, get_inner_model
from rl4mixed.gurobi.vrp import solve as solve_w_gurobi
from rl4mixed.heuristics.improvement import ImprovementHeuristic
from rl4mixed.active_search import active_search
from rl4mixed.models.attention_model import BaseAttnModel


log = logging.getLogger(__name__)


def get_validation_score(actor: BaseAttnModel, data: DataLoader, decode_type: str, **kwargs):

    actor.eval()
    set_decode_type(actor, decode_type, **kwargs)

    def eval_model_bat(bat):
        with torch.no_grad():
            _, _, state = actor(bat.to(actor.device))
            cost = state.get_final_cost()
        return cost.mean().item()

    rewards = [
        eval_model_bat(bat)
        for bat
        in tqdm(data)
    ]

    return np.mean(rewards)



class ValidationHandler:

    def __init__(self,
                 test_params: opt.TestParams,
                 instance_params: opt.InstanceParams,
                 instance_dir: str,
                 actor=None,
                 neptune_client=None,
                 device=None) -> None:

        self.actor = actor

        self.neptune_client = neptune_client or neptune.init_run(mode="debug")

        self.instance_params = instance_params
        self.test_params = test_params

        self.decode_type = test_params.decode_type
        self.beam_width = test_params.beam_width

        self.device = device or "cpu"
        self.render = test_params.render

        self.grb_timeout = test_params.gurobi_timeout

        self.actor_val_reward = None
        self.rewards = defaultdict(list)
        self.num_not_exact = 0

        self.ran_validation = False

        self.dataset_path = os.path.join(instance_dir, "dataset.pth")

        assert self.decode_type != "beam_search" or self.beam_width is not None


    def _run_heuristic(self, instance: BatchedInstances) -> InstanceSolution:

        if instance.heur_obj is not None:
            heuristic_solution = instance.heur_obj
        else:
            # solve instance
            heuristic_solution = ImprovementHeuristic(instance, seed=self.instance_params.seed).solve()

        # save reward
        self.rewards["heuristic"].append(heuristic_solution.reward)
        self.neptune_client[f"test/compare/runtimes/heuristic"].append(heuristic_solution.runtime)
        self.neptune_client[f"test/compare/distances/heuristic"].append(heuristic_solution.reward)

        return heuristic_solution

    
    def _run_gurobi(self, instance: BatchedInstances) -> InstanceSolution:
        if instance.exact_obj is not None:
            solution = instance.exact_obj
        else:
            # solve instance
            solution, _ = solve_w_gurobi(instance, verbose=False, timeout=self.grb_timeout)

        if solution.reward is None:
            # need a way to handle cases where gurobi does not come up with a solution
            self.rewards["gurobi"].append(np.nan)
            self.neptune_client[f"test/compare/distances/gurobi"].append(-.0)
        else:
            self.rewards["gurobi"].append(solution.reward)
            self.neptune_client[f"test/compare/distances/gurobi"].append(solution.reward)

        self.neptune_client[f"test/compare/runtimes/gurobi"].append(solution.runtime)
        return solution

    def _run_actor(self, batch: BatchedInstances) -> InstanceSolution:

        def _get_flat_idx(state):
            batch_ids = []
            for b_id, b in enumerate(state):
                flat_shelf_ids = []
                visited = b.visited_[1:].to("cpu")
                skus = b.sku_[1:].to("cpu")
                for shelf, sku in zip(visited, skus):
                    flat_shelf_ids.append(get_flat_ids(shelf, sku, batch[b_id]))

                batch_ids.append(flat_shelf_ids)

            return torch.Tensor(batch_ids)
            
        assert len(batch) == 1

        batch = batch.to(self.device)

        if self.test_params.active_search_iterations > 0:
            log.info("Performing active search...")
            self.actor = active_search(self.actor, batch, self.test_params)

        self.actor.eval()
        set_decode_type(self.actor, self.decode_type, beam_width=self.test_params.comp_beam_width)

        with torch.no_grad():
            start_time = time.time()
            trajectories, _, state = self.actor(batch)
            actor_runtime = time.time() - start_time

        if state.flat:
            # tour_indices = trajectories.shelf
            tour_indices = state.visited_[:,1:]
        else:
            # we need to remap the nested ids to the flat ids in order to compare solutions
            # tour_indices = _get_flat_idx(trajectories)
            tour_indices = _get_flat_idx(state)
        
        instance_reward = state.get_final_cost().to("cpu").numpy().tolist()[0]

        # te    
        units_w_depot = torch.cat((torch.zeros_like(state.taken_units[:, :1, :]), state.taken_units), 1)
        units = units_w_depot.gather(1, trajectories.shelf[...,None].expand(batch.bs, -1, batch.num_items))

        if batch.is_normalized:
            units = (units * batch.original_capacity).int()

        tours_and_units = get_full_tour_information(tour_indices, units)[0]

        self.actor.train()

        self.neptune_client[f"test/compare/distances/actor-{self.decode_type}"].append(instance_reward)
        self.neptune_client[f"test/compare/runtimes/actor-{self.decode_type}/"].append(actor_runtime)
        self.rewards["actor"].append(instance_reward)
        return InstanceSolution(instance_reward, tours_and_units, runtime=actor_runtime)


    def write_neptune(self):
        self.neptune_client[f"test/compare/summary"] = {
            "gurobi": np.mean(self.rewards["gurobi"]), 
            f"actor-{self.decode_type}": np.mean(self.rewards["actor"]),
            "heuristic": np.mean(self.rewards["heuristic"]),
            "gurobi-no-solution": self.num_not_exact
        }

        if self.num_not_exact > 0:
            # mask instances with no excakt solution from comparison
            mask = [i == np.nan for i in self.rewards["gurobi"]]
            masked_rewards = {}
            for k,v in self.rewards.items():
                masked_rewards[k] = np.ma.array(v, mask=mask)

            self.neptune_client[f"test/compare/summary/masked"] = {
                "gurobi": np.mean(masked_rewards["gurobi"]), 
                f"actor-{self.decode_type}": np.mean(masked_rewards["actor"]),
                "heuristic": np.mean(masked_rewards["heuristic"]),
            }

    def val_actor(self, data):

        val_reward = get_validation_score(self.actor, 
                                          data, 
                                          self.decode_type, 
                                          beam_width=self.test_params.beam_width)

        self.neptune_client[f"test/{self.decode_type}/test_reward"] = val_reward


    def compare(self, data: List[BatchedInstances]):

        original_actor_state_dict = get_inner_model(self.actor).state_dict()

        for cnt, instance in enumerate(data):

            assert len(instance) == 1
            assert instance.exact_obj is not None
            assert instance.heur_obj is not None

            actor_solutions = self._run_actor(instance)  
            get_inner_model(self.actor).load_state_dict(original_actor_state_dict)

            exact_solution = self._run_gurobi(instance[0])

            heuristic_solution = self._run_heuristic(instance[0])

            if self.render:

                if actor_solutions.tour_and_units is not None:
                    fig_actor = render_solution(instance, actor_solutions)
                    self.neptune_client[f"test/compare/solution_plots/{cnt}-actor-{self.decode_type}"].upload(fig_actor)

                if exact_solution.tour_and_units is not None:
                    fig_opt = render_solution(instance, exact_solution)
                    self.neptune_client[f"test/compare/solution_plots/{cnt}-gurobi"].upload(fig_opt)

                if heuristic_solution.tour_and_units is not None:
                    fig_heur = render_solution(instance, heuristic_solution)
                    self.neptune_client[f"test/compare/solution_plots/{cnt}-heuristic"].upload(fig_heur)

            if exact_solution.reward is None:
                self.num_not_exact += 1

        self.neptune_client[f"test/{self.decode_type}/compare"]

        

    def evaluate(self):
        log.info("Looking for test dataset in %s" % self.dataset_path)
        if os.path.exists(self.dataset_path):
            log.info("Found existing test dataset!")
            test_data, comp_data = torch.load(self.dataset_path)
        else:
            log.info("Could not find existing test dataset...")
            test_data, comp_data = self.gen_eval_dataset(save=not self.instance_params.debug)

        self.val_actor(test_data)
        if self.test_params.n_exact_instances > 0:
            self.compare(comp_data)
            self.write_neptune()

            log.info("Comparison Done: \n")
            log.info("Exact average reward: %s. Actor average reward: %s. Heuristic average reward: %s" % (
                np.mean(self.rewards["gurobi"]),
                np.mean(self.rewards["actor"]),
                np.mean(self.rewards["heuristic"])
            ))

        self.ran_validation = True

    
    def gen_eval_dataset(self, save=True):

        log.info("...start generating dataset for instance type %s" % self.instance_params.id)
        self.neptune_client["id"] = self.instance_params.id

        test_data = MixedShelvesDataset(self.test_params.test_size, self.test_params.batch_size_test, self.instance_params)
        test_loader = DataLoader(test_data, batch_size=None, batch_sampler=None, shuffle=False)

        comp_data = MixedShelvesDataset(self.test_params.n_exact_instances, batch_size=1, instance_params=self.instance_params)
        comp_loader: List[BatchedInstances] = DataLoader(comp_data, batch_size=None, batch_sampler=None, shuffle=False)
        comp_dataset: List[BatchedInstances] = []


        for i, instance in enumerate(comp_loader):

            log.info("Generate exact solutions for %sth instance with id %s" % (i+1, self.instance_params.id))
            exact_solution, is_opt = solve_w_gurobi(instance[0], timeout=self.test_params.gurobi_timeout, add_to_timeout=self.test_params.no_sol_add_timeout)
            
            log.info("Generate heurstic solutions for %sth instance with id %s" % (i+1, self.instance_params.id))
            heuristic_solution = ImprovementHeuristic(instance[0], seed=self.instance_params.seed).solve()  

            if exact_solution.tour_and_units is not None:
                fig_opt = render_solution(instance, exact_solution)
                self.neptune_client[f"plots/{i}-gurobi"].upload(fig_opt)

            if heuristic_solution.tour_and_units is not None:
                fig_heur = render_solution(instance, heuristic_solution)
                self.neptune_client[f"plots/{i}-heuristic"].upload(fig_heur)

            instance.exact_obj = exact_solution
            instance.heur_obj = heuristic_solution

            comp_dataset.append(instance)

            self.neptune_client["data/num_finished"] = i+1
            self.neptune_client["data/found_optimum"].append(int(is_opt))
            self.neptune_client["data/gurobi_runtimes"].append(exact_solution.runtime)
            self.neptune_client["data/gurobi_obj"].append(exact_solution.reward or -0.)
            self.neptune_client["data/heuristic_obj"].append(heuristic_solution.reward)
            self.neptune_client["data/heuristic_runtime"].append(heuristic_solution.runtime)

        combined_dataset = (test_loader, comp_dataset)

        if save:
            torch.save(combined_dataset, self.dataset_path)
            self.neptune_client[f"data/dataset"].upload(self.dataset_path)

        log.info("Finished generating dataset for instance type %s" % self.instance_params.id)
        return combined_dataset


@hydra.main(version_base=None, config_path="configs/", config_name="sim-test-data")
def main(cfg: DictConfig):
    from hydra.core.hydra_config import HydraConfig
    hc = HydraConfig.get()
    instance_dir = hc.runtime.output_dir

    instance_params = opt.InstanceParams(**cfg.instance)
    test_params = opt.TestParams(**cfg.test)


    log.info("write dataset into directory %s" % instance_dir)

    if instance_params.debug:
        run = neptune.init_run(mode="debug")
    else:
        neptune_user = os.getenv("NEPTUNE_USER")
        run = neptune.init_run(project=f"{neptune_user}/rl4mspr-datagen")

    run["status/code"] = 0

    val_handler = ValidationHandler(test_params=test_params, 
                                    instance_params=instance_params, 
                                    instance_dir=instance_dir, 
                                    neptune_client=run)
    
    try:
        val_handler.gen_eval_dataset(save=not instance_params.debug)

    except Exception as e:
        run["status/code"] = 1
        run["status/error"] = str(e)
        raise e

    run.stop()

if __name__ == "__main__":
    main()