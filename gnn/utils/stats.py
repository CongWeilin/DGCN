
import os
import time
import yaml
import copy
import numpy as np

class Stats(object):
    
    def __init__(self, config):

        self.config = config

        # print(self.config.dataset)

        # Loss and Scores
        self.train_loss = []
        self.train_scores = []

        self.val_loss = []
        self.val_scores = []

        self.test_score = 0

        # best model
        self.best_model = []
        self.best_val_score = 0
        self.best_val_loss = 1e10
        self.best_val_epoch = 0

        # Timing info
        self.train_time_epochs = []

        # Gradient
        self.per_epoch_grads = [] # This can take lots of space
        
    @property
    def run_id(self):
        current_counter = 1

        if os.path.exists(self.config.output_dir):
            for fn in os.listdir(self.config.output_dir):
                if fn.startswith(self.config.run_name) and fn.endswith('npz'):
                    current_counter += 1
                
        return '{}-{:03d}'.format(self.config.run_name, current_counter)

    @property
    def run_output(self):
        output = os.path.join(self.config.output_dir, self.run_id)
        return output

    
    def save(self):
        config_vars = vars(self.config)

        stats_vars = copy.copy(vars(self))
        stats_vars.pop('config', None)

        # create output folder
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)
        
        # save model to torch TODO: later
        # remove from stats
        stats_vars.pop('best_model', None)
        
        # save config and stats to npy
        np.savez(self.run_output, config=config_vars, stats=stats_vars)
    
        return config_vars, stats_vars

    @staticmethod
    def load(stats_file):
        all_data = np.load(stats_file, allow_pickle=True)
        config = all_data['config'][()]
        stats = all_data['stats'][()]
        return config, stats

    @property
    def best_train_time(self):
        return np.sum(self.train_time_epochs[:self.best_val_epoch])

    @property
    def total_train_time(self):
        return np.sum(self.train_time_epochs)