import os
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import rl4mixed.settings as opt
from rl4mixed.problems.dataset import MixedShelvesDataset
from rl4mixed.gurobi.vrp import solve


@hydra.main(version_base=None, config_path="../configs/", config_name="config")
def main(cfg: DictConfig):

    instance_params = opt.InstanceParams(**cfg.instance)
    test_params = opt.TestParams(**cfg.test)


    # torch.manual_seed(instance_params.seed)

    test_data = MixedShelvesDataset(test_params.test_size,
                                    test_params.batch_size_test,
                                    instance_params)
                
    test_loader = DataLoader(test_data, batch_size=None, batch_sampler=None)

    cnt = 0
    for batch in test_loader:
        batch = batch.flatten().unnormalize_batch()
        for instance in batch:
            solve(instance, mipfocus=True)
            cnt += 1

if __name__ == "__main__":

    main()


