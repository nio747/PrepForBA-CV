## Visualizing the loss function

- Slicing through the high-dimensional space along rays or along planes (2 dimensions).
- We can generate a random direction $W_1$ and compute the loss along this direction by $L(W+aW_1)$ for different values of $a$. Two-dimensional: $L(W+aW_1+bW_2)$ with $a$, $b$ as x-axis and the y-axis. 



## Optimization

Goal: find $W$ that minimizes the loss function. 

### #1: bad idea -- random search

simply try out many different random weights and keep track of what works best. 

```python
bestloss = float("inf") # Python assigns the highest possible float value
for num in xrange(1000):
	W = np.random.randn(10,3073)*0.0001 #generate random parameters
	loss = L(X_train, Y_train, W)
	if loss < bestloss: 
		bestloss = loss
		bestW = W
	print `in attemp %d the loss was %f， best %f` % (num, loss, bestloss)
```

**Iterative refinement**: make it slightly better each time.



### #2: Random Local Search

take a step only if it leads downhill(when $W + \delta W$ is lower, then we will perform an update)

### #3: Following the Gradient

#### Computing the gradient:

- **numerical gradient** with finite differences:

  ```python
  def eval_numerical_gradient(f,x):
  	# f should be a function that takes a single argument
  	# x is the point (np array) to evaluate the gradient at
  	
  	fx=f(x) # evaluate function value at original point
  	grad = np.zeros(x.shape)
  	h = 0.00001
  	
      # iterate over all indexes in x
  	it = np.ndite(x, flags=[`multi_index],op_flags=[`readwrite`])
  	while not it.finished:
  	
      	# evaluate f(x+h)
  		ix = it.multi_index
  		old_value = x[ix]
  		x[ix] = old_value + h # increment by h
  		fxh = f(x) # evaluate f(x+h)
  		x[ix] = old_value # restore to prev. value (important)
  		
          # compute the partial derivative
  		grad[ix] = (fxh - fx)/h # slope
  		it.iternext() # step to next dimension
  	
  	return grad
  ```

  **Practical considerations**: better using centered difference formula: $[f(x+h) - f(x-h)] / 2 h$

  Compute the gradient of the CIFAR-10 loss function at some random point in the weight space:

  ```python
  def CIFAR_loss_fun(W):
  	return L(X_train, Y_train, W)

  W = np.random.rand(10, 3073)*0.001 # random weight vector
  df = eval_numerical_gradient(CIFAR_loss_fun, W) # get the gradient
  ```

  make an update:

  ```python
  loss_original = CIFAR_loss_fun(W) # the original loss
  print `original loss: %f` % (loss_original,)

  for step_size_log in [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]:
  	step_size = 10 ** step_size_log
  	W_new = W - step_size * df # new position in the weight space
  	loss_new = CIFAR_loss_fun(W_new)
  	print `for step size %f new loss: %f` % (step_size, loss_new)
  ```

  **Effect of step size**: how far along this direction we should step. Step size = *learning rate*, a very important yet headache-inducing hyperparameter in NN.  \\*overstep*

  **problem of efficiency**: complexity linear in the number of parameters.

- **analytic gradient**: The numerical gradient is simple but it is approximate and it is very computationally expensive. Analytical way: it can be more error prone, in practice: very common to compute the analytic gradient and compare it to the numerical gradient to check the correctness of your implementation. --> **gradient check**

  ​

## Gradient Descent

The procedure of repeatedly evaluating the gradient and then performing a parameter update. 

```python
#vanilla version
while True:
	weights_grad = evaluate_grad(loss_fun,data,weights)
	weights += - step_size * weights_grad # parameter update
```



**Mini-batch gradient descent**. In large-scale applications: wasteful to compute the full loss function over the entire training set to perform only a single parameter update. A very common approach: compute the gradient over batches of the training data. (e.g. ConvNet, a typical batch contains 256 examples from 1.2 million examples.)

```python
# vanilla
while True:
	data_batch = sample_training_data(data,256) # sample 256 examples 
	weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
	weights += -step_size * weights_grad # perform update
```

The gradient from a mini-batch is a good approximation of the gradient of the full objective. Therefore, much faster convergence can be achieved by evaluating the mini-batch gradients to perform more frequent parameter updates. 

Extreme case: mini-batch contains only a single example: **Stochastic Gradient Descent(SGD)** or **on-line** gradient descent.  The size of the mini-batch is a hyperparameter but it is not very common to cross-validate it. It is usually based on memory constraints (if any), or set to powers of 2.