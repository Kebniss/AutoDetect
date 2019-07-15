import torch
from torch import nn
from torch.utils.data import DataLoader

from apex import amp

from ignite.engine import Events, Engine, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, RunningAverage, Precision, Recall
from ignite.contrib.handlers import ProgressBar, CosineAnnealingScheduler
from ignite.utils import convert_tensor


class Trainer(Engine):
    def __init__(self,
                 train_ds,
                 test_ds,
                 model,
                 criterion,
                 experiment_name,
                 checkpoint_root="/checkpoints",
                 device=None,
                 non_blocking=False,
                 amp_opt_level=None,
                 prepare_batch_fn=None,
                 output_transform=lambda x, y, y_pred, loss: loss.item(),
                 cycle_mult=0.9):

        self.train_ds = train_ds
        self.test_ds = test_ds
        self.model = model
        self.criterion = criterion
        self.experiment_name = experiment_name
        self.checkpoint_root = checkpoint_root
        self.device = device
        self.non_blocking = non_blocking
        self.amp_opt_level = amp_opt_level
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-2)  # placeholder lr
        self.optimizer.zero_grad()
        if self.amp_opt_level:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=self.amp_opt_level)

        self.max_gradient_norm = 1.0  # placeholder
        self.scheduler = None
        self.prepare_batch_fn = prepare_batch_fn or self.prepare_batch
        self.output_transform = output_transform
        self.cycle_mult = cycle_mult

        if device:
            model.to(self.device)

        self.epochs_run = 0
        self.epoch_state = {}

        super().__init__(self.process_fn)

    def prepare_batch(self, batch):
        """Prepare batch for training: pass to a device with options.
        """
        x, y = batch
        return (convert_tensor(
            x, device=self.device, non_blocking=self.non_blocking),
                convert_tensor(
                    y, device=self.device, non_blocking=self.non_blocking))

    def process_fn(self, engine, batch):
        self.model.train()
        x, y = self.prepare_batch_fn(batch)
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y) / self.gradient_accumulation_steps
        if self.amp_opt_level:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if engine.state.iteration % self.gradient_accumulation_steps == 0:
            if self.amp_opt_level:
                nn.utils.clip_grad_norm_(
                    amp.master_params(self.optimizer),
                    self.max_gradient_norm,
                    norm_type=2)
            else:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_gradient_norm,
                    norm_type=2)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return self.output_transform(x, y, y_pred, loss)

    def train(self,
              lr,
              batch_size,
              n_epochs,
              gradient_accumulation_steps=1,
              num_workers=0,
              max_gradient_norm=1.0):
        self.optimizer.param_groups[0]['lr'] = lr
        self.max_gradient_norm = max_gradient_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True)
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False)

        if self.scheduler:
            self.remove_event_handler(self.scheduler, Events.ITERATION_STARTED)
        self.scheduler = CosineAnnealingScheduler(
            self.optimizer.param_groups[0],
            'lr',
            lr / 10,
            lr,
            len(self.train_loader),
            start_value_mult=self.cycle_mult,
            end_value_mult=self.cycle_mult,
        )
        self.add_event_handler(Events.ITERATION_STARTED, self.scheduler, "lr")

        metrics_dict = {
            'accuracy': Accuracy(),
            'nll': Loss(self.criterion),
            'precision': Precision().tolist()[1],
            'recall': Recall().tolist()[1],
        }
        metrics_dict[
            'f1'] = metrics_dict['precision'] * metrics_dict['recall'] * 2 / (
                metrics_dict['precision'] + metrics_dict['recall'])

        self.evaluator = create_supervised_evaluator(
            self.model, device=self.device, metrics=metrics_dict)
        self.attach_common_handlers()

        self.run(self.train_loader, max_epochs=n_epochs)

    def attach_common_handlers(self):
        RunningAverage(output_transform=lambda x: x).attach(self, 'loss')

        pbar = ProgressBar()
        pbar.attach(self, ['loss'])

        @self.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            self.evaluator.run(self.test_loader)
            metrics = self.evaluator.state.metrics
            self.epochs_run += 1
            self.epoch_state[self.epochs_run] = metrics
            print(f"Validation Results - Epoch: {self.state.epoch}: "
                  f"Avg accuracy: {metrics['accuracy']:.2f} |"
                  f"Precision: {metrics['precision']:.2f} |"
                  f"Recall: {metrics['recall']:.2f} | "
                  f"F1: {metrics['f1']:.2f} | "
                  f"Avg loss: {metrics['nll']:.2f}")

        epoch_checkpointer = ModelCheckpoint(
            self.checkpoint_root,
            filename_prefix=self.experiment_name,
            score_name="f1",
            score_function=lambda _: self.evaluator.state.metrics['f1'],
            n_saved=5,
            atomic=True,
            create_dir=True,
            require_empty=False,
        )
        self.evaluator.add_event_handler(
            Events.EPOCH_COMPLETED, epoch_checkpointer,
            {self.model.__class__.__name__: self.model})
