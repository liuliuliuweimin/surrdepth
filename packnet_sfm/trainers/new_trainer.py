import os
import torch
from packnet_sfm.trainers.base_trainer import BaseTrainer, sample_to_cuda
from packnet_sfm.utils.config import prep_logger_and_checkpoint
from packnet_sfm.utils.logging import print_config
from packnet_sfm.utils.logging import AvgMeter

from torch.utils.tensorboard import SummaryWriter


idx_log = 0


class NewTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
        if torch.cuda.is_available():
            torch.cuda.set_device("cuda:0")
        else:
            torch.cuda.set_device("cpu")
        torch.backends.cudnn.benchmark = True

        self.avg_loss = AvgMeter(50)
        self.dtype = kwargs.get("dtype", None)  # just for test for now

        self.writer = SummaryWriter()

    @property
    def proc_rank(self):
        return 1

    @property
    def world_size(self):
        return 1

    def fit(self, module):

        # Prepare module for training
        module.trainer = self
        # Update and print module configuration
        prep_logger_and_checkpoint(module)
        print_config(module.config)

        # Send module to GPU
        module = module.to('cuda')
        # Configure optimizer and scheduler
        module.configure_optimizers()
        # Create distributed optimizer
        # compression = hvd.Compression.none
        optimizer = module.optimizer
        scheduler = module.scheduler

        # Get train and val dataloaders
        train_dataloader = module.train_dataloader()
        val_dataloaders = module.val_dataloader()



        # Epoch loop
        for epoch in range(module.current_epoch, self.max_epochs):
            # Train
            metrics = self.train(train_dataloader, module, optimizer)
            avg_train_loss = metrics['avg_train-loss']
            avg_train_photometric_loss = metrics['avg_train-photometric_loss']
            avg_train_smoothness_loss = metrics['avg_train-smoothness_loss']
            avg_train_supervised_loss = metrics['avg_train-supervised_loss']


            # writer.add_scalar('Loss/avg_train_loss'
            #                   , avg_train_loss, epoch)
            # writer.add_scalar('Loss/avg_train_photometric_loss'
            #                   , avg_train_photometric_loss, epoch)
            # writer.add_scalar('Loss/avg_train_smoothness_loss'
            #                   , avg_train_smoothness_loss, epoch)
            # writer.add_scalar('Loss/avg_train_supervised_loss'
            #                   , avg_train_supervised_loss, epoch)


            # Validation
            validation_output = self.validate(val_dataloaders, module)
            # Check and save model
            self.check_and_save(module, validation_output)
            # Update current epoch
            module.current_epoch += 1
            # Take a scheduler step
            scheduler.step()

    def train(self, dataloader, module, optimizer):
        global idx_log
        # Set module to train
        module.train()
        # Shuffle dataloader sampler
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(module.current_epoch)
        # Prepare progress bar
        progress_bar = self.train_progress_bar(
            dataloader, module.config.datasets.train)
        # Start training loop
        outputs = []
        # For all batches
        for i, batch in progress_bar:
            # Reset optimizer
            optimizer.zero_grad()
            # Send samples to GPU and take a training step
            batch = sample_to_cuda(batch)
            output = module.training_step(batch, i)
            # Backprop through loss and take an optimizer step
            output['loss'].backward()
            optimizer.step()
            # Append output to list of outputs
            output['loss'] = output['loss'].detach()
            outputs.append(output)
            # Update progress bar if in rank 0
            if self.is_rank_0:
                progress_bar.set_description(
                    'Epoch {} | Avg.Loss {:.4f}'.format(
                        module.current_epoch, self.avg_loss(output['loss'].item())))
            if i % 20 == 0 and i > 0:
                idx_log += 1
                metrics = module.training_epoch_end(outputs[-20:])
                avg_train_loss = metrics['avg_train-loss']
                avg_train_photometric_loss = metrics['avg_train-photometric_loss']
                avg_train_smoothness_loss = metrics['avg_train-smoothness_loss']
                avg_train_supervised_loss = metrics['avg_train-supervised_loss']
                print('##new_trainer',i," ",idx_log)
                # idx = (module.current_epoch+1) * 48 + i
                self.writer.add_scalar('Loss/avg_train_loss'
                                  , avg_train_loss, idx_log)
                self.writer.add_scalar('Loss/avg_train_photometric_loss'
                                  , avg_train_photometric_loss, idx_log)
                self.writer.add_scalar('Loss/avg_train_smoothness_loss'
                                  , avg_train_smoothness_loss, idx_log)
                self.writer.add_scalar('Loss/avg_train_supervised_loss'
                                  , avg_train_supervised_loss, idx_log)



        # Return outputs for epoch end
        return module.training_epoch_end(outputs)

    def validate(self, dataloaders, module):
        # Set module to eval
        module.eval()
        # Start validation loop
        all_outputs = []
        # For all validation datasets
        # print("dataloaders length:",len(dataloaders))
        for n, dataloader in enumerate(dataloaders):
            # Prepare progress bar for that dataset
            progress_bar = self.val_progress_bar(
                dataloader, module.config.datasets.validation, n)
            outputs = []
            # For all batches
            for i, batch in progress_bar:
                # Send batch to GPU and take a validation step
                batch = sample_to_cuda(batch)
                output = module.validation_step(batch, i, n)
                # Append output to list of outputs
                outputs.append(output)
            # Append dataset outputs to list of all outputs
            all_outputs.append(outputs)
        # Return all outputs for epoch end
        return module.validation_epoch_end(all_outputs)

    def test(self, module):
        # Send module to GPU
        module = module.to('cuda', dtype=self.dtype)
        # Get test dataloaders
        test_dataloaders = module.test_dataloader()
        # Run evaluation
        self.evaluate(test_dataloaders, module)

    @torch.no_grad()
    def evaluate(self, dataloaders, module):
        # Set module to eval
        module.eval()
        # Start evaluation loop
        all_outputs = []
        # For all test datasets
        for n, dataloader in enumerate(dataloaders):
            # Prepare progress bar for that dataset
            progress_bar = self.val_progress_bar(
                dataloader, module.config.datasets.test, n)
            outputs = []
            # For all batches
            for i, batch in progress_bar:
                # Send batch to GPU and take a test step
                batch = sample_to_cuda(batch, self.dtype)
                output = module.test_step(batch, i, n)
                # Append output to list of outputs
                outputs.append(output)
            # Append dataset outputs to list of all outputs
            all_outputs.append(outputs)
        # Return all outputs for epoch end
        return module.test_epoch_end(all_outputs)
