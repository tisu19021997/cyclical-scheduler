import tensorflow.keras.backend as K
from tensorflow import keras
import matplotlib.pyplot as plt


class _BaseScheduler(keras.callbacks.Callback):
    def __init__(self,
                 max_lr,
                 base_lr=None,
                 div_factor=20,
                 cyclical_momentum=True,
                 base_momentum=0.85,
                 max_momentum=0.95):

        """Base class for cyclical learning rate scheduler.

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
        """
        super().__init__()

        if not max_lr and not base_lr:
            raise ValueError('Either max_lr or base_lr must be defined.')
        elif max_lr and base_lr:
            self.max_lr = max_lr
            self.base_lr = base_lr
        elif max_lr:
            self.max_lr = max_lr
            self.base_lr = max_lr / div_factor
        else:
            self.max_lr = base_lr * div_factor
            self.base_lr = base_lr

        # It is called "epochCounter" in the original papper. However,
        # it increases after each batch ends, therefore, I call it batch_counter.
        self.batch_counter = 0
        self.train_iterations = 0
        self.cyclical_momentum = cyclical_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        self.history = {}

        self._reset()

    def _reset(self):
        # Reset the batch counter.
        self.batch_counter = 0

    def get_momentum(self):
        raise NotImplementedError

    def get_lr(self):
        raise NotImplementedError

    def plot_lr(self, figsize=(10, 3)):
        if not self.train_iterations or not self.history:
            raise ValueError('Nothing to plot. Train the model first.')

        totals = 1
        cols = 1
        y = [self.history['lrs']]
        x = self.history['iterations']
        ylabels = ['Learning Rate']

        if self.cyclical_momentum:
            totals += 1
            cols += 1
            y.append(self.history['moms'])
            ylabels.append('Momentum')

        position = range(1, totals + 1)
        fig = plt.figure(figsize=figsize)

        for i in range(totals):
            ax = fig.add_subplot(1, cols, position[i])
            ax.plot(x, y[i])
            ax.set_xlabel('Iteration')
            ax.set_ylabel(ylabels[i])

        fig.tight_layout(pad=3.0)
        plt.show()

    def plot_train_accr_lr(self):
        if not self.batch_counter or not self.history:
            raise ValueError('Nothing to plot. Train the model first.')

        accs = self.history['accuracy']
        lrs = self.history['lrs']

        # max_y = np.max(accs)
        # max_x = np.max(lrs)
        # min_x = np.min(lrs)

        plt.title('Train Accuracy vs Learning Rate')
        plt.plot(lrs, accs)
        # plt.plot([min_x, max_x], [max_y, max_y], 'r--')
        plt.xlabel('Learning Rate')
        plt.ylabel('Train Accuracy')

    def _check_support(self):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute to use cyclical learning rate.')

        if self.cyclical_momentum and not hasattr(self.model.optimizer, 'momentum'):
            raise ValueError('Optimizer must have a "momentum" attribute to use cyclical momentum.')

    def on_train_begin(self, logs={}):
        self._check_support()
        logs = logs or {}

        if self.batch_counter == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)

            if self.cyclical_momentum:
                K.set_value(self.model.optimizer.momentum, self.max_momentum)
        else:
            K.set_value(self.model.optimizer.lr, self.get_lr())

            if self.cyclical_momentum:
                K.set_value(self.model.optimizer.momentum, self.get_momentum())

    def on_batch_end(self, epoch, logs={}):
        self._check_support()
        logs = logs or {}

        # Increase the epoch counter.
        self.batch_counter += 1
        self.train_iterations += 1

        # Keep track of the learning rates.
        self.history.setdefault('lrs', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.train_iterations)
        K.set_value(self.model.optimizer.lr, self.get_lr())

        # Also keep track of accuracy.
        for key, value in logs.items():
            self.history.setdefault(key, []).append(value)

        # Keep track of momentums if supported.
        if self.cyclical_momentum:
            self.history.setdefault('moms', []).append(K.get_value(self.model.optimizer.momentum))
            K.set_value(self.model.optimizer.momentum, self.get_momentum())
