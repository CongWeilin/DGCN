from . import Full

class Batch(Full):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # selecting sampler and init
        # selecting sampler for validation and inference (if any)
        # * Add support for sampling inference

        self.train_loader = None

        
        
        self.val_loader = None
        self.test_loader = None
        if not self.config.exact_infer:
            pass

    def run(self):

        self.tbar = self.trange(self.config.num_epochs, desc='Training Epochs')

        # if all features can fit on device, load them once

        for epoch in self.trange:
            # needs additional sampling and movement here

            self.train(epoch)

            self.validation(epoch)

        self.inference()

    def train(self, epoch):
        
        # Forward
        self.model.train()

        # Backward
        
        # update stats

        return None

    def validation(self, epoch):

        if full_validation:
            return super().validation(args)
        else:
            pass

    def inference(self):
        # same as validtion
        pass