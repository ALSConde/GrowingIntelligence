import time
import json
from avalanche.training.plugins import SupervisedPlugin

class DataLoadingTimePlugin(SupervisedPlugin):
    def __init__(self):
        self.current_exp_id = None
        self.batch_times = []
        self.exp_times = {}  
        self._last_end_time = None

    def before_training_exp(self, strategy, **kwargs):
        self.current_exp_id = strategy.clock.train_exp_counter
        self.batch_times = []
        self._last_end_time = None

    def before_training_iteration(self, strategy, **kwargs):
        now = time.perf_counter()
        if self._last_end_time is not None:
            self.batch_times.append(now - self._last_end_time)

    def after_training_iteration(self, strategy, **kwargs):
        self._last_end_time = time.perf_counter()

    def after_training_exp(self, strategy, **kwargs):
        if self.batch_times:
            self.exp_times[self.current_exp_id] = {
                "mean_batch_loading_time_ms":
                    1000.0 * sum(self.batch_times) / len(self.batch_times),
                "num_batches": len(self.batch_times),
            }
        else:
            self.exp_times[self.current_exp_id] = {
                "mean_batch_loading_time_ms": None,
                "num_batches": 0,
            }

    def get_results(self):
        return self.exp_times
