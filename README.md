# Cyclical LR and 1-Cycle Policy as Keras's callback.

An online Google Colab notebook is available [here](https://colab.research.google.com/drive/1mD0aYIPoVLC-GeJvNwK4rwM3I5v3_qzZ?usp=sharing).

**Me**: https://github.com/tisu19021997

**Preferences**:

*   Leslie N.Smith, ["Cyclical Learning Rates for Training Neural Networks"][CLR] (2015)
*   Leslie N.Smith, ["A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay"][2018] (2018)
*   Frank Hutter et al., ["SGDR: Stochastic Gradient Descent with warm restarts"][SGDR] (2017) 

**Related works**:

* [bckenstler/CLR](https://github.com/bckenstler/CLR)
* [PyTorch implementation of OneCycleLR](https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR)

**TODO**:
  - [x] Cyclical Momentum
  - [ ] Learning Rate finder (similar to the one that [fastai][fastai_lr_finder] implemented)
  - [x] ~~Consine Annealing (like [PyTorch's][PyTorch])~~
  - [ ] Unit test

[fastai_lr_finder]: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
[PyTorch]: https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
[CLR]: https://arxiv.org/abs/1506.01186
[2018]: https://arxiv.org/abs/1803.09820
[SGDR]: https://arxiv.org/abs/1608.03983