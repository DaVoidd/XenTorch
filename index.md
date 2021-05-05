## API Manual

### About

XenTorch is a module which has a goal of simplifying Neural Networks and AI development in Roblox. As you may have already noticed, _XenTorch_ is a name derived from the Python module _PyTorch_. This is for implification of the fact that using this module is similar to PyTorch, although there are quite a few differences. If you have programmed neural networks in Python before, you should be able to adapt to this module faster than others.

### NOTICE

This module has only been newly released. Therefore, there may be various bugs and errors and it is recommended to not completely rely on this and have a back-up, especially when dealing with big networks.

*Note: Square barackets are used in some places instead of curly barackets, this is done in order to prevent errors with the HTML.*

### Get Started

The following example is a demonstration of a neural network which will learn to predict an output based on the given training examples.
For demonstration purposes, the training data has a rule associated with it. If the 2nd input is a 1 or a 0 the output should be {0, 0, 1} and {0, 0, 0} respectively.
After training finishes, the last 4 lines of code will display what the network predicts, given 2 seen and 2 unseen data inputs, so you can see how it performs. Once you have the code ready, click 'Run' to start the training process.
\
\
**Model Layout: 3 x 3/No bias -> ReLU -> 3 x 3/Bias -> Sigmoid** | Total of 9 nodes
\
\
**Optimizer:** Algorithm used for training\
**lr:** Learning Rate \
**termination:** The threshold of error rate at which training ceases

```markdown
local Example_Network_1 = XenTorch.nn.Sequential({
	XenTorch.nn.Linear(3, 3, false),
	{Activation = {XenTorch.nn.ReLU, XenTorch.nn.Prime.ReLU}},
	XenTorch.nn.Linear(3, 3, true),
	{Activation = {XenTorch.nn.Sigmoid, XenTorch.nn.Prime.Sigmoid}}
}, {XenTorch.nn.Cost.MSE, XenTorch.nn.Prime.Cost.MSE})

local x_data = { {0, 1, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1} }
local y_data = { {0, 0, 1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 1}, {0, 0, 1} }

local train_set, test_set = XenTorch.Data.Separate(x_data, y_data, 1)

local x_train, y_train = table.unpack(train_set)
local x_test, y_test = table.unpack(test_set)

local lr = 1
local termination = 0.0015
local Optimizer = 'GD'

Example_Network_1 = XenTorch.Network.FitData(Example_Network_1, x_train, y_train, Optimizer, lr, x_test, y_test, termination)

print(XenTorch.Network.Run(Example_Network_1, {1, 0, 1}))
print(XenTorch.Network.Run(Example_Network_1, {0, 1, 0}))
print(XenTorch.Network.Run(Example_Network_1, {2, 0, 5}))
print(XenTorch.Network.Run(Example_Network_1, {0, 1, 3}))
```

## Model Functions


### XenTorch.nn.Sequential(model, cost_function)

Generates a new network.

Model format:
```markdown
{mandatory: layer, optional: layer/activation, ...}
```

Layer format:
```markdown
layer_function()
```
or you may use the following array format where the rows are input nodes and columns are the output nodes
```markdown
3 x 2 Layer
[[w11, w12],
 [w21, w22],
 [w31, w32]]
```

Activation function format:
```markdown
{activation_function, activation_function_derivative}
```

Cost function format
```markdown
{cost_function, cost_function_derivative}
```

Mind the paranthesis syntax for the functions. Layers have them, activation and cost functions do not.

### XenTorch.nn.Linear(input_dim, output_dim, bias:false)
Creates a linear layer of dimensions `input_dim * output_dim`. If `bias`is set to true, all biases will be initialized to 0. Or if preferred, a number can be assigned as the inital biases. `bias = false` means the layer will not be assigned biases.

### XenTorch.nn.Wise(function)
Makes functions that require a single input compatible with the network. You may use this with custom activation functions. However, if the function is dependent on the other nodes in the layer, like Softmax, then you would need to code it yourself to be compatible with layer inputs. Derivates of input nodes with respect to output nodes at different indices is currently not supported, but you may still use the derivatives of input nodes with respect to the output nodes at the same indices.

### XenTorch.nn.Intellect(function)
Makes functions that require 2 inputs and return 1 output compatible with the network. Used for Cost functions where the first input is the prediction and the second is the correct label.

### XenTorch.nn.ReLU(x), XenTorch.nn.Prime.ReLU(x)
Rectified Linear Unit activation function and its derivative.

### XenTorch.nn.Sigmoid(x), XenTorch.nn.Prime.Sigmoid(x)
Sigmoid activation function and its derivative.

### XenTorch.nn.Softmax(x), XenTorch.nn.Prime.Softmax(x)
Softmax activation function and its derivative. Please note, the Softmax derivative function currently **does not take into account derivatives of input nodes with respect to output nodes at different indices**, therefore training with Softmax may not be as fast, although it still works.

### XenTorch.nn.Cost.MSE(y, y_hat), XenTorch.nn.Prime.Cost.MSE(y, y_hat)
Mean Squared Error function and its derivative.

## Training Functions


### XenTorch.Data.Separate(x_labels, y_labels, batch_size, ordered:false, validation:false)
Splits data into `batch_size` sized groups and returns training and test sets. If ordered is set to false, the data will be randomized. If validation is set to true, the function will also return a validation set.

#### **Ratios**
Training:Test = 75:25\
Training:Test:Validation = 70:15:15
