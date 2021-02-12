

class Config(object):
    def __init__(self, config):
        
        self.dataset = ''
        self.output_dir = ''
        self.run_name = 'result'

        self.model = ''
        self.layer = ''
        self.activation = ''
        self.num_layers = ''
        self.hidden_dim = ''
        
        self.prop = ''
        self.arch = ''

        self.loss = ''
        self.optim = ''

        self.gpu = ''
        self.num_epochs = ''
        self.lr = ''
        self.dropout = ''
        self.wd = ''
        
        self.adj_norm = ''
        self.input_norm = True
        self.layer_norm = True

        self.exact_infer = True
        self.exact_val = True
        self.create_split = ''

        self.patience = 0
        self.val_steps = 1
        
        # Samplings
        self.num_neighbors = ''
        self.drop_ratio = ''

        # model specific params
        self.alpha = 0.2
        self.beta = 0.5
        
        # experimental
        self.fully_connected = False
        
        for key, value in config.items():
            setattr(self, key, value)

        if self.dataset is not None and type(self.dataset) != str:
            self.dataset = self.dataset.name

    def __repr__(self):

        all_config = vars(self)
        for c in all_config:
            print(c, all_config[c])
        
        return ""