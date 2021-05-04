## API Manual


### About

This module has only been newly released. Therefore, there may be various bugs/errors and it is recommended to not completely rely on this and have a back-up, especially when dealing with big networks.

### Get Started

The following example is a demonstration of a neural network which will learn how to predict an output based on the given training examples.
For example sake, the training data has a rule associated with it. If the 2nd input is a 1, the output is {0, 0, 1} and if the 2nd input is a 0, then the output is {0, 0, 0}.
After training finishes, the last 2 lines of code will display what the network predicts, given unseen data.
\
\
**Model Layout: 3 x 3/No bias -> ReLU -> 3 x 3/Bias -> Sigmoid** | Total of 9 nodes
\
\
**Optimizer:** Algorithm in charge of training\
**lr:** Learning Rate \
**termination:** The threshold of error rate at which training ceases

![image](https://user-images.githubusercontent.com/65914250/116998669-3f0a6200-ace7-11eb-97b8-8e6bec6ba135.png)
