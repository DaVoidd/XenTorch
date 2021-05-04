## API Manual


### About


### Get Started

```markdown
`
-- 3 x 3 x 3 Linear Model, ReLU hidden activation and Sigmoid output activation, output layer bias initialized to 0

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

local lr = 1 --Learning rate
local termination = 0.0015 --The error rate at which we want to stop training

Example_Network_1 = XenTorch.Network.FitData(Example_Network_1, x_train, y_train, 'GD', lr, x_test, y_test, termination) --Full training sequence with normal Gradient Descent

print(XenTorch.Network.Run(Example_Network_1, {2, 0, 5})) --Random data to see the output; correct output should be 0, 0, 0
print(XenTorch.Network.Run(Example_Network_1, {0, 1, 3})) --correct output: 0, 0, 1
`
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/DaVoidd/XenTorch/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
