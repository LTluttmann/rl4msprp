import numpy as np
from torch.utils.data import DataLoader
from rl4mixed.settings import get_args, ModelParams, TestParams
from rl4mixed.problems.dataset import MixedShelvesDataset
from rl4mixed.heuristics.improvement import ImprovementHeuristic


if __name__ == "__main__":

    args = get_args()
    model_params: ModelParams = args.model
    test_params: TestParams = args.test


    test_data = MixedShelvesDataset(test_params.test_size,
                                    test_params.batch_size_test,
                                    model_params)
                
    test_loader = DataLoader(test_data, batch_size=None, batch_sampler=None)

    cnt = 0
    rewards = []
    for batch in test_loader:
        for instance in batch:
            sol = ImprovementHeuristic(instance, 1234567).solve()
            cnt += 1
            rewards.append(sol.reward)

            if cnt >= test_params.n_exact_instances:
                break
        if cnt >= test_params.n_exact_instances:
            break
    print(np.mean(rewards))
    
