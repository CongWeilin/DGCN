
import sys
import numpy as np

sys.path.append('..')
from gnn import datasets, layers, models, training, utils

def run_repeated(num_runs, config, dataset):
    all_test_score = []
    for rep in range(num_runs):
        print(f'run {rep} of {num_runs}')
        test_score = run_experiment(config, dataset)
        print('-'*40)
        all_test_score.append(test_score)

    all_test_score = np.array(all_test_score) * 100
    print(f'{all_test_score.mean():.2f} ± {all_test_score.std():.2f}')
    print('#'*40)


def run_experiment(config, dataset):

    run_config = utils.Config(config)
    trainer = training.Full(run_config, dataset)
    print(trainer.model)

    trainer.run()
    trainer.save()

    return trainer.stats.test_score