Overview
========

Guiding Design Principles
-------------------------
A number of important requirement emerged out of the tfutils' redesign:

- **Simplicity:**  PTUtils supports a streamlined API and reduces the relience on specific conventions requrired to specify and run an experiment by enforcing an extremely concise set of rules on a single base class.

- **Modularity:** PTUtils increases fine-grain control over an experiment by generating a library of small, independent and *compatible* units of functionality from the same base class.

- **Flexibility:** expand set of possible usage patterns by specifying experiments in terms of the composition of customized functional units.

- **Extensibility:** PTUtils provides more opportunities for custom behavior by allowing users to override the default functionality of any given unit without disrupting the others.


Creating dynamic experiments
----------------------------
The 'define-by-run' paradigm established by deep learning frameworks such as Chainer, DyNet, PyTorch offers a powerful new way to structure neural network computations: the execution of the model/graph is conditioned on the state of the model itself, forming a *dynamic graph*. PTUtils attempts to **leverage and extend PyTorch's dynamic nature** by giving researchers a fully *dynamic experiment* whereby execution of the entire experiment is conditioned on the state of the experiment and any component contained within. The long-term motivation behind this approach is to provide researchers with a dynamic environment in which they can control the behavior/interactions of/between a model/collection of models as well as the data that is presented to the models, while saving the evolution of the environment behind the scenes.