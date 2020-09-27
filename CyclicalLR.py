import numpy as np
from BaseScheduler import _BaseScheduler


class CyclicalLR(_BaseScheduler):
    def __init__(self,
                 max_lr,
                 base_lr=None,
                 div_factor=20,
                 cyclical_momentum=True,
                 base_momentum=0.85,
                 max_momentum=0.95,
                 step_size=2000.0,
                 policy='triangular',
                 gamma=1.0,
                 scale_mode='cycle',
                 scaler=None):
        """Sets the learning rate according to the cyclical learning rate policy (CLR)
        mentioned in `Cyclical Learning Rates for Training Neural Networks`_. The policy varies
        the learning rate (in some cases, and the momentum) between the upper bound and
        the lower bound. Note that the policy updates the learning rate after very batch.

        There are 3 CLR policies implemented in this class:
          1. "triangular": A triangular cycle without amplitude scaling.
          2. "triangular2": The same as the "triangular" except the learning rate difference
          is cut in half at the end of each cycle.
          3. "exp_range": The boundaries are declined by an exponential factor of :math:`\text{gamma}^{\text{cycle iterations}}`

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
          step_size (float): The number of training iterations in half a cycle. Since, cycle=step_size*2
            Default: 2000.0
          policy (string): One of {triangular, triangular2, exp_range}. Each comes with a different scaler.
            If scaler is defined, this argument is ignored.
            Default: 'triangular'
          gamma (float): Constant used in 'exp_range' scaler.
            Default: 1.0
          scale_mode (str): One of {'cycle', 'iterations'}. Scaling mode of the scaler.
            Default: 'cycle'
          scaler (function): Custom scaler. Notice that the result of the scaler should be between 0 and 1.
            Default: None.

          .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
        """
        super().__init__(max_lr, base_lr, div_factor, cyclical_momentum, base_momentum, max_momentum)

        if policy not in ['triangular', 'triangular2', 'exp_range']:
            raise ValueError('Supported policies are "triangular", "triangular2", and "exp_range"')

        self.step_size = step_size
        self.policy = policy
        self.gamma = gamma

        if not scaler:
            if self.policy == 'triangular':
                self.scaler = self._triangular_scaler
                self.scale_mode = 'cycle'
            elif self.policy == 'triangular2':
                self.scaler = self._triangular2_scaler
                self.scale_mode = 'cycle'
            else:
                self.scaler = self._exp_range_scaler
                self.scale_mode = 'iterations'
        else:
            self.scaler = scaler
            self.scale_mode = scale_mode

    def _triangular_scaler(self, x):
        return 1

    def _triangular2_scaler(self, x):
        return 1.0 / (2.0 ** (x - 1))

    def _exp_range_scaler(self, x):
        return self.gamma ** x

    def get_lr(self):
        # Calculate the current cycle.
        cycle = np.floor(1 + self.batch_counter / (2 * self.step_size))

        # x will be negative so (1-x) will be positive on the first half of a cycle, therefore, the learning rate
        # will go up. On the other hand, it will be positive so (1-x) will be negative on the second half of a cycle,
        # makes the learning rate go down.
        x = np.abs(1 + self.batch_counter / self.step_size - 2 * cycle)

        # Update the learning rate.
        lr = (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))

        if self.scale_mode == 'cycle':
            lr *= self.scaler(cycle)
        else:
            lr *= self.scaler(self.batch_counter)

        return self.base_lr + lr

    def get_momentum(self):
        cycle = np.floor(1 + self.batch_counter / (2 * self.step_size))
        x = np.abs(1 + self.batch_counter / self.step_size - 2 * cycle)
        momentum = (self.max_momentum - self.base_momentum) * max(0, (1 - x))

        if self.scale_mode == 'cycle':
            momentum *= self.scaler(cycle)
        else:
            momentum *= self.scaler(self.batch_counter)

        return self.max_momentum - momentum
