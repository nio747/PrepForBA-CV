# CS231n-neural Networks, Andrej Karpathy


[TOC]
## Quick Intro
- Linear classification uses $s=Wx$ to compute scores for different categories given the image:
  ○ $W$: 10*3072 matrix;
  ○ $x$: 3072*1 input column vector containing all pixel data.
  ○ Output: 10*1 vector for 10 class scores (CIFAR-10)
- An Example neural network $s=W_2max(0,W_1x)$
    ○ $W_1$: 100*3072 Matrix transforming the image into a 100-dim vector.
    ○ Max(0,-): elementwise NL function. Thresholding to zero.
    ○ $W_2$: 10*100.
    ○ $W_1$, $W_2$ are learned with stochastic gradient descent. Their Gradients are derived with chain rule.
- A three Layer neural network $s=W_3max(0,W_2max(0,W_1x))$ 
    ○ $W_1​$, $W_2​$,$W_3​$ are parameters to be learned. 
    ○ The sizes of the intermediate hidden vectors are hyperparameters.
- [links-2](http://cs231n.github.io/neural-networks-2/)

## Modeling one neuron
### Biological connections
neuron, synapses, dendrites, axon. 
Input signals travel along the axons **$x_0$** interact with the dendrites of the other neuron based on the synapic strength **$w_0$**. When the final sum is above a certain threshold...neuron would be *fire*. We model the firing rate with an **activation function $f$**, representing the frequency of the spikes. Historically: sigmoid function $\sigma$.
![single neuron](http://cs231n.github.io/assets/nn1/neuron_model.jpeg)

``` 
class Neuron(object):
  # ... 
  def forward(self, inputs):
    """ assume inputs and weights are 1-D numpy arrays and bias is a number """
    cell_body_sum = np.sum(inputs * self.weights) + self.bias
    firing_rate = 1.0 / (1.0 + math.exp(-cell_body_sum)) # sigmoid activation function
    return firing_rate
```
### single neuron as a linear classifier
**Binary Softmax classifier**. interpret $\sigma(\sum_iw_ix_i + b)$ to be the prapability of one of the classes to formulate the cross-entropy loss and optimizing it would bead to a binary Softmax classifier(wheter the output is greater than 0.5).
**Binary SVM classifier**. Attach a max-margin hinge loss to the output and train it to become a binary SVM.
**Regularization interpretation**. The regularization loss interpreted as *gradual forgetting*(driving all weights $w$ towards zero.)

### commonly used activation functions
Every $f$ takes a single number and performs a fixed math. operation on it. 

**Sigmoid** $\sigma(\sum_iw_ix_i + b)$. restrict the input into range between 0 and 1. 

* *saturate and kill gradients.*
* *outputs are not zero-centered.* if the data coming in is always positive, then the gradient on the weights become either all be positive, or all negative, introducing zig-zagging dynamics in the gradient updates for the weights.

**Tanh**. [-1,1]; The tanh neuron is simply a scaled sigmoid neuron: $\tanh(x) = 2 \sigma(2x) -1$. It saturates, but is zero-centered. It is preffered to the $\sigma$.

**ReLU**. $\max(w_1^Tx+b_1, w_2^Tx + b_2)$ with $w_1, b_1 = 0$.  Aka. the Rectified Linear Unit. $f(x) = \max(0, x)$. Thresholding at zero. 
- (+) accelerate the convergence of gradient descent. 
- (+) non-expensive operation (compare to exponentials etc.)
- (-) can be fragile and *die* since they can get knocked off the data manifold. Neurons will never activate across the entire training dataset if the learning rate is set too high. 

**Leaky ReLU** for x<0: small negative slope. 

$f(x) = \mathbb{1}(x < 0) (\alpha x) + \mathbb{1}(x>=0) (x)$. Slope $\alpha$ can also be made into a parameter of each neuron --> PReLu. 

**Maxout** $\max(w_1^Tx+b_1, w_2^Tx + b_2)$. however too many paramters. 

**TLDR** 
- Use ReLu. Be careful with learning rates and monitor the fraction of dead units. 
- Give Leaky ReLu and Maxout a try.
- Never use sigmoid.
- Try tanh. worse than ReLu and Maxout.

##Architectures

###Layer-wise organization



> - Cycles are not allowed, otherwise infinite loop.
> - most common layer type: **fully-connected layer**(note: neurons within a single layer share no connections)

**Naming conventions**. N-layer: input layer doesn't count. 
**output layer** most commonly don't have an activation function. representing class scores in classification or real-valued target in regression.
**Sizing** using number of neurons(not counting the inputs), or more commonly the number of *learnable* parameters. 
Modern ConvNet contain 100 million paramters and are usually made up of 10-20 layes.(Note: with parameter sharing)

### A feed-forward computation
Organized into layers: simple and efficient using matrix vector operations. All connection strengths for a layer can be stored in a single matrix. 
``` python
# forward-pass of a 3-layer neural network:
f=lambda x: 1.0/(1.0 + np.exp(-x)) #sigmoid activation function
x=np.random.randn(3,1) #random input of three numbers as a 3*1 vector
h1=f(np.dot(W1,x)+b1) # 4*3-3*1-+4*1 --- 4*1 first hidden layer
h2=f(np.dot(W2,x)+b2) # second
out=np.dot(W3,h2)+b3 # output neuron (1*4 4*1 1*1)--(1*1)
```
$W_i, b_i$ are the learnable parameters. 



### Representational power

> - The neural network can approximate any continuous function (**universal function approximators**).
> - Deeper networks work better than a single-hidden-layer networks, despite equal representational power.
> - In stark contrast to ConvNet, going deeper than 3 Layers rarely helps. (Images contain hierarchical structure)



### Setting number of layers and sizes

- As we increase the size and number of layers, the **capacity** increases. When larger, more complicated functions. (easier **overfitting** when a model fits the noise in the data). Smaller: better **generalization**, but hard to train using local methods like Gradient Descent. *Their loss functions have fewer local minima with high loss (easier to converge to).* 
- Using **regularization strength** to control the overfitting.  Bigger $\lambda$ ! 
- Use as big as possible (based on your budget), and other regularization techniques to control overfitting.





## Setting up the data and the model

### Data Preprocessing

3 common forms of data preprocessing for a data matrix $X$ of size $N*D$ . N: number of data; D: their dimensionality. 

**Mean subtraction**: most common; subtracting the mean across every individual *feature* in the data, has the geometric interpretation of centering the cloud of data around the origin along every dimension.

 `X-=np.mean(X, axis=0)`  ; for images: (single value from all pixels): `X -=np.mean(X)` or separately across three color channels.



**Normalization**: normalizing the dimensions so that approx. of  same scale. 

- Either divide each dimension by its standard deviation, if have already been **zero-centered**.   

    `X /=np.std(X,axis=0)` 

- Or normalizes each dim. within [-1,1]. 

- Only make sense if different features have different scales (units) but are equally important to the algorithm. 

- As for Images, the relative scales of pixels are approx. equal(0-255), so not strictly necessary.



**PCA and Whitening**: first centered as above. then compute the covariance matrix to get the correlation structure in the data.

```python
X-=np.mean(X, axis=0) #zero centered!
cov = np.dot(X.T, X)/X.shape[0] #get covariance matrix
```

(i,j) element = covariance between i- and j-th dimension of the data. Diagonal = variances. Symmetric and positive semi-definite.

To compute the SVD factorization of the Covariance Matrix:

```python
U,S,V=np.linalg.svd(cov)
```

columns of `U`: eigenvectors, sorted by their eigenvalues; `S`: a 1-D array of the singular values. 

To decorrelate, project the original, zero-centered data into the eigenbasis:

```python
Xrot = np.dot(X,U) #actually a rotation of the data so that the new axes are the eigenvectors.
```

Since in `U` the eigenvectors are sorted, reduce dimensionality using only the top few eigenvectors and discarding the dimensions along which the data has no variance (**Principal Component Analysis**) .

```python
Xrot_reduced = np.dot(X, U[:,:100]) #[N x 100], keeping 100 dim. of the data that contain most variance. saving both space and time.
```



The **whitening** operation takes the data in the eigenbasis, divides every dimension by eigenvalue to normalize the scale. If input is a multivariable Gaussian, then the whitened data will be a gaussian with zero mean and identity covariance matrix. Geometrically = Stretching and squeezing the data into an isotropic gaussian blob.

```python
Xwhite = Xrot/np.sqrt(S+1e-5) #divide by the eigenvalues which are square roots of the singular values, adding 1e-5 to prevent division by zero.
```

Note: *Exaggerating noise*. when adding 1e-5 it can greatly exaggerate the noise in the data, since it stretches all dimensions including the irrelevant dimensions to be of equal size in the input. Can be mitigated by stronger smoothing (e.g. increasing 1e-5) .

![preprocessing](http://cs231n.github.io/assets/nn2/prepro2.jpeg)



> PCA/Whitening are not used with ConvNet. However, it is very important to zero-center the data, and it is common to normalize every pixel.



-------------------------------------------------------------------------------

### Weight Initialization 

Before we can begin to train the network we have to initialize its parameter.

- ~~Pitfall: all zero initialization.~~ It is reasonable to assume: half of the weights are positive. All zeros: no source of asymmetry between neurons if their weights are initialized to be the same --- same gradients, same parameter updates.

- **Small random number**: We still want $W_i$ to be close to 0. *Symmetry breaking*. `W = 0.01* np.random.randn(D,H)` , `randn` samples from a zero mean, unit std deviation gaussian. 

  > It's not the case that smaller will work better: very small weights compute very small gradients, leading to diminishing the 'gradient signal'.

- **Calibrating the variances with $\sqrt{1/n}$**: 

  > one problem with Small random number: the distribution of the output has a variance that grows with the number of inputs (**fan-in**)

  We can normalize the variance of each neuron's output to 1 by scaling its $W_i$ by $\sqrt{fan-in}$ : 	 `w=np.random.randn(n)/sqrt(n)` , `n` is the number of its inputs. So that all neurons initially have approx same output distribution, improve the rate of convergence.  

  > Note: ReLU units will have a positive mean inputs and weights. 

  > current recommendation with ReLU neurons: `w = np.random.randn(n)*sqrt(2.0/n)` 

- **Sparse initialization**: set all weight matrices to $0$, but to break symmetry every neuron is randomly connected to a fixed number of neurons below it. **???**

- **Initializing the biases**. Possible and common to be $0$! For ReLU: some like to use small constant as $0.01$ for all bias. This ensures all ReLU units fire in the beginning, and obtain and propagate some gradient. But not proved. more common to simply use 0.

- **In practice**,  current recommendation is to use ReLU and `w = np.random.randn(n) * sqrt(2.0/n).`

- **Batch Normalization**, *Insert BatchNorm layer after FC Layers or ConvLayers and before NL.* very **common** practice! More robust to bad intialization. can be interpreted as doing preprocessing at every layer of the network. 

-----------------------------

### Regularization

control the capacity of NN to prevent overfitting

**L2 regularization**: most common. Penalizing the  squared magnitude of all parameters $\frac{1}{2} \lambda w^2$  directly in the objective. $\lambda$ - **regularization strength**. Interpretation: heavily penalizing peaky weight vectors, preferring diffuse weight vectors (use all of its inputs a little rather that some a lot). During gradient descent update: ultimately L2 means every weight is decayed linearly: `w+= -lambda*w` towards $0$.

**L1 Regularization**: for each $w$ add $\lambda\mid w \mid$ . Combine L1 and L2 (Elastic net regularization):  $\lambda_1 \mid w \mid + \lambda_2 w^2$. *Neurons with L1*: using only subset of most important inputs and become nearly invariant to the noise. Final $W_i$ with L2 are usually diffuse, small numbers, in practice superior, when not connecting to explicit feature selection.

**Max norm constraints**：using projected gradient descent to enforce a constraint(upper bond) on $ \Vert \vec{w} \Vert_2$. In practice: parameter update as normal, enforcing $\Vert \vec{w} \Vert_2 < c$ for every neuron. **$c$** are typical on orders of 3 or 4. The Network would not "explode" even learning rates are too high. 

**Dropout**: extremely effective, simple. While training, only keeping a neuron active with probability $P$ (hyperparameter) or setting it to Zero otherwise. During testing no dropout.

Vanilla Dropout: not recommended.

**Inverted Dropout**: recommended, drop and scale at train time, doin't do anything at test time. 

```python
p=0.5 #probability of keeping a unit active. higher p means less dropout.

def train_step(X): #forward pass for 3-layer nn
	H1=np.maximum(0,np.dot(W1,x)+b1) #hidden layer 1
	U1=(np.random.rand(*H1.shape)<p)/p #first dropout mask, with p! 
    H1*=U1 #drop.
    H2 = np.maximum(0, np.dot(W2, H1) + b2)
    U2 = (np.random.rand(*H2.shape) < p) / p # second dropout mask. Notice /p!
    H2 *= U2 # drop!
    out = np.dot(W3,H2)+b3
    
#backward: gradient ...
#parameter update...

def predict(X):
    H1=np.maximum(0,np.dot(W1,X)+b1) # no scaling!!
    H1=np.maximum(0,np.dot(W2,H1)+b2)
    out = np.dot(W3,H2)+b3
```

**Noise**: Dropout introduces stochastic behavior in the forward pass. During test, the noise is marginalized over analytically or numerically. see *DropConnect*. 

**Bias regularization**: not common but in practice would not leads to worse performance.

**Pre-layer regularization** not very common to regularize different layers to different amounts. 

**In practice**: most common -- a single, global L2 regularization strength that is cross-validated.   also common -- combine with dropout applied after all layers, $p=0.5$ by default, can be tuned on validation data. 



### Loss functions

*Data loss* measures the compatibility between a prediction and the ground truth label, takes an average over data losses for every individual example. 

**Classification**: A dataset of examples and a single correct label. 

$$L_i = \sum_{j\neq y_i} \max(0, f_j - f_{y_i} + 1)$$ or with squared hinge loss $\max(0, f_j - f_{y_i} + 1)^2$.

Or Softmax classifier using cross-entropy loss $$L_i = -\log\left(\frac{e^{f_{y_i}}}{ \sum_j e^{f_j} }\right)$$

**Attribute classification**. When $y_i$ is a binary vector, where every example may or may not have a certain attribute and where the attributes are not exclusive. 

$$L_i = -\log\left(\frac{e^{f_{y_i}}}{ \sum_j e^{f_j} }\right)$$ or train a logistic regression classifier for every attribute independently: binary logistic regression classifier. 

**Regression** see lecture notes. 



## Learning

### Gradient Check

Comparing the analytic gradient$f’_a$ to the numerical gradient$f’_n$. 

**Use the centered formula**: 

$$\frac{df(x)}{dx} = \frac{f(x + h) - f(x - h)}{2h} \hspace{0.1in} \text{(use instead)}$$ Evaluate the loss function twice, error terms on order $O(h^2)$ 

**Compare the relative error**: $\frac{\mid f'_a - f'_n \mid}{\max(\mid f'_a \mid, \mid f'_n \mid)}$ (use max or add both terms). Must explicitly keep track of the case where both are zero and pass the gradient check in that edge case. In practice:

- .>1e-2: the gradient is probably wrong
- 1e-2 ~ 1e-4: uncomfortable
- <1e-4: usually okay for objectives with kinks. But when no kinks (using tanh and softmax), too high.
- 1e-7 and less: happy.

Deeper network means higher relative errors. 1e-2 might be okay for 10-layer network. 

**Use double precision**. 

**Stick around active range of floating point**: read through this [paper](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html). *Print the raw n/a gradients, make sure the numbers are not extremely small, if so, scale up by a constant to a nicer range where float are more dense, ideally on order of 1.0*

**Kinks in the objective**, Kinks = non-differentiable parts of an objective function, introduced by $f$ such as ReLU, SVM loss, Maxout etc. E.g. for ReLU, $x<0$: analytic gradient is zero, numerical gradient might be non-zero, since $f(x+h)$ might cross over the kink. 

By keeping track of the identities of all winners in a function $max(x,y)$ when evaluating $f(x+h)$ and then $f(x-h)$: when the identity changed, a kink was crossed, and the $f'_n$ will not be exact.

**Use only few datapoints.**

**Be careful with the step size h**. When h is much smaller -- numerical precision problems. Sometimes when the gradient doesn't check, possible change $h$ to be 1e-4 or 1e-6. See Wikipedia [Numerical differentiation](https://en.wikipedia.org/wiki/Numerical_differentiation).

**Gradcheck during a chara. mode of operation**: correct at one point != correct globally. Best to use a short **burn-in** time during which the network is allowed to learn and perform the gradient check after the loss starts to go down. Don't perform it at the 1. iteration. Otherwise pathological edge case, mask an incorrect implementation of the gradient. 

**Don't let the regularization (loss) overwhelm the data (loss)**. in which case the gradients will be primarily coming from the regularization term. Recommended: turn off regularization, check data loss alone first, then the regularization term second and independently (by removing data loss contribution in the code **or** increasing the regularization strength $\lambda$ so that its effect is non-negligible and an incorrect implementation would be spotted. 

**Turn off dropout/augmentations**. Turn off any non-deterministic effects in the network, such as dropout, random data augmentations... Downside: you wouldn't be gradient checking them. A better solution: force a particular random seed before evaluation $f(x+h)$, $f(x-h)$ and $f'_a$.

**Check only few dimensions**.  In practice: million parameters. only practical to check some dimensions of the gradient and assume others are correct. **Make sure** to gradcheck a few dimensions for every separate parameter. 



### Sanity Checks before learning

**Look for correct loss at chance performance**. make sure to get the expected loss when initializing with small parameters. best to first check the data loss alone (setting $\lambda$ = $0$) For CIFAR-10 with softmax: 2.302, for weston watkins SVM a loss of 9. 

As second sanity check, increasing $\lambda$ should increase the loss.

**Overfit a tiny subset** before training on the full dataset to achieve zero cost. also best to set $\lambda = 0$. Note: it may happen that can overfit very small dataset but still have an incorrect implementation!



### Babysitting the learning process

#### Loss function

#### Train/Val accuracy

#### Ratio of weights:updates

#### Activation/Gradient distributions per layer

#### First-Layer Visualizations



### Parameter updates

#### SGD, bells, whistles

- Vanilla update: 
- Momentum update:
- Nesterov Momentum

#### Annealing the learning rate

- Step decay
- Exponential decay
- 1/t decay

#### Second order methods

#### Per-parameter adaptive learning rate methods

- Adagrad
- RMSprop
- Adam

### Hyperparameter optimization

**Implementation**.

**Prefer one validation fold to cross-validation**

**hyperparameter ranges**

**Prefer random search to grid search**

**Careful with best values on border**

**Stage search from coarse to fine**

**Bayesian Hyperparam. Optim**

### Evaluation

#### Model Ensembles

- same model, different initializations
- Top models discovered during cross-validation
- different checkpoints of a single model
- running average of parameters during training
- ​

### Summary 

1. Gradcheck with small batch of data, be aware of pitfalls.
2. Sanity check: make sure initial loss is reasonable, and 100% accuracy on a very small portion of data
3. During the training: monitor the loss, train/val accuracy, (magnitude of updates ~ parameter values (about 1e-3)) first layer weights (when using ConvNet)
4. Two recommended updates: SGD+Nesterov, or Adam
5. Decay learning rate over the period of training. e.g. halve it after x epochs, or whenever the validation accuracy tops off.
6. Search for good hyperparam with random search \ ~~grid search~~. Stage search from coarse (wide hyperparam ranges, only for 1-5 epochs) to fine (narrower, training for many more epochs)
7. form model enselmbles for extra performence. 









