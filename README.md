# PTUtils
[![Development Status](https://img.shields.io/badge/development%20status-alpha-brightgreen.svg)](https://github.com/alexandonian/ptutils/blob/master)
[![Build Status](https://travis-ci.org/alexandonian/ptutils.svg?branch=master)](https://travis-ci.org/alexandonian/ptutils)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/alexandonian/ptutils/blob/master/LICENSE.txt)
[![Latest Stable Version](https://img.shields.io/badge/stable-N%2FA-yellow.svg)](https://github.com/alexandonian/ptutils/blob/master)
[![Latest Unstable Version](https://img.shields.io/badge/unstable-dev-orange.svg)](https://github.com/alexandonian/ptutils/blob/master)

PyTorch utilities for neural network research.


## You have just found PTUtils.
PTUtils is a PyTorch utility package designed for coordinating neural network experiments with dynamics and modularity in mind. Inspired by its Tensorflow sibling, [tfutils](https://github.com/neuroailab/tfutils), PTUtils provides functionality for:

- constructing, runing and monitoring dynamic neural network models
- retrieving data from multiple data sources interactively
- interfacing with common databases

Read the documentation at [PTUils.io](https://github.com/alexandonian/ptutils).

PTUtils is compatible with: __Python 2.7__.

------------------

### Guiding Design Principles
A number of important requirement emerged out of the tfutils' redesign:

* __Simplicity:__  PTUtils supports a streamlined API and reduces the relience on specific conventions requrired to specify and run an experiment by enforcing an extremely concise set of rules on a single base class.

* __Modularity:__ PTUtils increases fine-grain control over an experiment by generating a library of small, independent and *compatible* units of functionality from the same base class.

* __Flexibility:__ expand set of possible usage patterns by specifying experiments in terms of the composition of customized functional units.

* __Extensibility:__ PTUtils provides more opportunities for custom behavior by allowing users to override the default functionality of any given unit without disrupting the others.

**Acknowledgments:** PTUtils' design was inspired from numerous sources/frameworks:

| General  | Tensorflow | PyTorch          | Misc.      |
| :------- |:---------  | :--------------- | :-----     |
| Keras    | TFUtils    | Torch.utils      | Django     |
| Teras    | TFLearn    | pytorch-examples | Flask      |
| Kur      | TF-Slim    | pytorch-tutorial | Requests   |
| Chainer  | Sonnet     | TorchNet         | Jsonpickle |
| Neon     | Polyaxon   | TorchSample      | Datastore  |
| Py2Learn | Baselines  | Inferno          | Scrapy     |

### Creating dynamic experiments
The 'define-by-run' paradigm established by deep learning frameworks such as Chainer, DyNet, PyTorch offers a powerful new way to structure neural network computations: the execution of the model/graph is conditioned on the state of the model itself, forming a *dynamic graph*. PTUtils attempts to **leverage and extend PyTorch's dynamic nature** by giving researchers a fully *dynamic experiment* whereby execution of the entire experiment is conditioned on the state of the experiment and any component contained within. The long-term motivation behind this approach is to provide researchers with a dynamic environment in which they can control the behavior/interactions of/between a model/collection of models as well as the data that is presented to the models, while saving the evolution of the environment behind the scenes. 


---

## Proposed Control Flow

The figure below depicts the intended high-level control flow of PTutils. Each module will operate independently as a standalone unit. You will be free to use any combination of modules that best suites your needs without worrying about inter-module dependencies. For example, you may choose to only use the DBInterface class for saving results to a database and handle the rest of the experiment yourself. Alternatively, you may choose to subclass `Config` and let PTUtils handle the rest. It's up to you!

![alt text](control_flow.png "Control Flow")

The details are explained below.

