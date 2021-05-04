## API Manual
*Note: Square barackets are used in some places instead of curly barackets, this is done in order to prevent errors with the HTML.*

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

```markdown
local Example_Network_1 = XenTorch.nn.Sequential({
	XenTorch.nn.Linear(3, 3, false),
	{Activation = {XenTorch.nn.ReLU, XenTorch.nn.Prime.ReLU}},
	XenTorch.nn.Linear(3, 3, true),
	{Activation = {XenTorch.nn.Sigmoid, XenTorch.nn.Prime.Sigmoid}}
}, {XenTorch.nn.Cost.MSE, XenTorch.nn.Prime.Cost.MSE})

local x_data = {{0, 1, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}}
local y_data = {{0, 0, 1}, {0, 0, 0}, {0, 0, 1}, {0, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 1}, {0, 0, 1}}

local train_set, test_set = XenTorch.Data.Separate(x_data, y_data, 1)

local x_train, y_train = table.unpack(train_set)
local x_test, y_test = table.unpack(test_set)

local lr = 1
local termination = 0.0015
local Optimizer = 'GD'

Example_Network_1 = XenTorch.Network.FitData(Example_Network_1, x_train, y_train, Optimizer, lr, x_test, y_test, termination)

print(XenTorch.Network.Run(Example_Network_1, {2, 0, 5}))
print(XenTorch.Network.Run(Example_Network_1, {0, 1, 3}))
```

## Model Functions


### XenTorch.nn.Sequential(model, cost_function)

Generates a new network.

Model format:
```markdown
[mandatory: layer, optional: layer/activation, ...]
```

Layer format:
```markdown
layer_function()
```
or you may use the following format where the rows are input nodes and columns are the output nodes
```markdown
3 x 2 Layer
[[w11, w12],
 [w21, w22],
 [w31, w32]]
```

Activation function format:
```markdown
[activation_function, activation_function_derivative]
```

Cost function format
```markdown
[cost_function, cost_function_derivative]
```

Mind the paranthesis syntax for the functions. Layers have them, activation and cost functions do not.

### XenTorch.nn.Linear(input_dim, output_dim, bias)
Creates a linear layer of dimensions `input_dim * output_dim`. If `bias`is set to true, all biases will be initialized to 0. Or if preferred, a number can be assigned as the inital biases. `bias = false` means the layer will not be assigned biases.

### XenTorch.nn.Wise(function)
Makes functions that require a single input compatible with the network.

### XenTorch.nn.Intellect(function)
Makes functions that require 2 inputs compatible with the network.

### XenTorch.nn.ReLU(x), XenTorch.nn.Prime.ReLU(x)
Rectified Linear Unit activation function and its derivative.

### XenTorch.nn.Sigmoid(x), XenTorch.nn.Prime.Sigmoid(x)
Sigmoid activation function and its derivative.

### XenTorch.nn.Softmax(x), XenTorch.nn.Prime.Softmax(x)
Softmax activation function and its derivative. Please note, the Softmax derivative function is currently **not a FULLY implemented function**, therefore training with Softmax may not be as fast, although it still works.
