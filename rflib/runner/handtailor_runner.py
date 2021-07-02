from rflib.runner.epoch_based_runner import EpochBasedRunner
from .builder import RUNNERS


@RUNNERS.register_module()
class HandTailorRunner(EpochBasedRunner):
    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer, self._epoch, **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, self._epoch, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
