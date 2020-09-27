import math
from CyclicalLR import CyclicalLR


class OneCycle(CyclicalLR):
    def __init__(self,
                 max_lr,
                 base_lr=None,
                 div_factor=20,
                 cyclical_momentum=True,
                 base_momentum=0.85,
                 max_momentum=0.95,
                 total_steps=None,
                 epochs=None,
                 steps_per_epoch=None,
                 min_div_factor=1e4,
                 inc_ratio=0.3,
                 anneal='linear'):
        """Anneals the learning rate according to the 1cycle learning rate policy. In short,
        the 1cycle policy anneals the learning rate from a base learning rate to the maximum learning rate and then
        from there to the minimum learning rate (which is lower than the the base learning rate according to the author).

        Args:
            max_lr (float): The upper learning rate bound in the cycle.
            base_lr (float): The lower learning rate bound or the initial learning rate in the cycle.
              If not provided, it is determined as max_lr/div_factor.
              Default: None
            div_factor (float or int): Determines the lower learning rate bound as max_lr/div_factor.
              Default: 20
            cyclical_momentum (bool): If ``True``, cyclical momentum will be used.
              Default: True
            base_momentum (float): The lower momentum bound or the initial momentum in the cycle.
              Default: 0.85
            max_momentum (float): The upper momentum bound in the cycle.
              Default: 0.95
            total_steps (int): The total number of steps in the cycle. If not provided,
              then it will be computed using epochs and steps_per_epoch.
            epochs (int): The number of epochs.
            steps_per_epoch (int): The number of steps per epoch. Typically,
              equals to ceil(num_samples / batch_size).
            min_div_factor (float): Determines the minimum learning rate bound as base_lr/min_div_factor.
              Default: 1e4
            inc_ratio (float): The ratio of the cycle for increasing the learning rate from 0 to 1.
              Default: 0.3
            anneal (str): Type of annealing function used. One of {linear, cosine, exp}.
              Default: 'linear'
        """

        super().__init__(max_lr, base_lr, div_factor, cyclical_momentum, base_momentum, max_momentum)

        if total_steps is None and epochs is None and steps_per_epoch is None:
            raise ValueError('Either total_steps or (epochs and steps_per_epoch) must be defined.')
        elif total_steps is not None:
            self.total_steps = total_steps
        else:
            self.total_steps = epochs * steps_per_epoch

        if inc_ratio < 0 or inc_ratio > 1:
            raise ValueError(f'Expected 0 <= inc_ratio <= 1, got {inc_ratio}.')

        # Compute number of steps the learning rate goes up and down.
        # self.step_size_up = float(inc_ratio * self.total_steps) - 1
        self.step_size_up = float(inc_ratio * self.total_steps) - 1
        self.step_size_down = float(self.total_steps - self.step_size_up) - 1

        # Since the author wanted to keep decreasing the learning rate for several last iterations,
        # the min_lr is the smallest value the learning rate can be.
        self.min_lr = self.base_lr / min_div_factor

        if anneal == 'linear':
            self.annealer = self._lin_annealing
        elif anneal == 'exp':
            self.annealer = self._exp_annealing
        else:
            self.annealer = self._cos_annealing

    def _exp_annealing(self, start, end, ratio):
        """Anneal exponentially from `start` to `end` as `ratio` goes from 0.0 to 1.0"""
        return start * (end / start) ** ratio

    def _cos_annealing(self, start, end, ratio):
        """Cosine anneal from `start` to `end` as `ratio` goes from 0.0 to 1.0"""
        return end + (start - end) / 2.0 * (math.cos(math.pi * ratio) + 1)

    def _lin_annealing(self, start, end, ratio):
        """Anneal linearly from `start` to `end` as `ratio` goes from 0.0 to 1.0"""
        return start + (end - start) * ratio

    def get_lr(self):
        step_num = self.batch_counter

        # If the number of total steps defined smaller than the current epoch counter,
        # the learning rate will be negative.
        if step_num > self.total_steps:
            raise ValueError(
                f'Tried to step {step_num} times. The specified number of total steps is {self.total_steps}')

        if step_num <= self.step_size_up:
            lr = self.annealer(self.base_lr, self.max_lr, step_num / self.step_size_up)
        else:
            down_step_num = step_num - self.step_size_up
            lr = self.annealer(self.max_lr, self.min_lr, down_step_num / self.step_size_down)

        return lr

    def get_momentum(self):
        step_num = self.batch_counter

        if step_num > self.total_steps:
            raise ValueError(
                f'Tried to step {step_num} times. The specified number of total steps is {self.total_steps}')

        if step_num <= self.step_size_up:
            momentum = self.annealer(self.max_momentum, self.base_momentum, step_num / self.step_size_up)
        else:
            down_step_num = step_num - self.step_size_up
            momentum = self.annealer(self.base_momentum, self.max_momentum, down_step_num / self.step_size_down)

        return momentum
