# Cyclical Learning Rate (CLR) and 1-Cycle Policy as Keras's callback.

An online Google Colab notebook is available [here](https://colab.research.google.com/drive/1mD0aYIPoVLC-GeJvNwK4rwM3I5v3_qzZ?usp=sharing).

**Me**: https://github.com/tisu19021997

## Short Description:
* In short, the Cyclical LR method allows the learning rate of a model to vary between two boundaries when training. 
By that, it provides substantial improvements in performance for different architectures. 
Cyclical LR divides the training phase into cycles and each cycle consists of 2 steps.
* **The 1-Cycle** policy uses the cyclical LR method but only with 1 cycle for the whole training. Moreover, 
this policy suggests that "always use one cycle that is smaller than the total number of iterations/epochs and
 allow the learning rate to decrease several orders of magnitude less than the initial learning rate for the remaining iterations".
* There are 2 variations of 1-Cycle policy that I found when doing my research:
    * In the first variation, the learning rate varies in 3 states: 
        1. from `base_lr` to `max_lr`
        2. from `max_lr` to `base_lr`
        3. from `base_lr` to `min_lr` (where `min_lr=base_lr/some_factor`)
    * In the second variation (which I am using here), the learning rate varies in 2 states:
        1. from `base_lr` to `max_lr`
        2. from `max_lr` to `min_lr`   

## Preferences

*   Leslie N.Smith, ["Cyclical Learning Rates for Training Neural Networks"][CLR] (2015)
*   Leslie N.Smith and Nicholay Topin ["Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"][super_convergence] (2017)
*   Leslie N.Smith, ["A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay"][2018] (2018)
*   Frank Hutter et al., ["SGDR: Stochastic Gradient Descent with Warm restarts"][SGDR] (2017) 

## Related Works

* [bckenstler/CLR](https://github.com/bckenstler/CLR)
* [PyTorch implementation of OneCycleLR](https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR)

## TODO
  - [x] Cyclical Momentum
  - [ ] Learning Rate finder (similar to the one that [fastai][fastai_lr_finder] implemented)
  - [x] ~~Consine Annealing (like [PyTorch's][PyTorch])~~
  - [ ] Unit test

[fastai_lr_finder]: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
[PyTorch]: https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
[CLR]: https://arxiv.org/abs/1506.01186
[super_convergence]: https://arxiv.org/abs/1708.07120
[2018]: https://arxiv.org/abs/1803.09820
[SGDR]: https://arxiv.org/abs/1608.03983