# CS213n-ConvNet, Andrei Karpathy

[TOC]



# Layer in ConvNet

 

## Overview:

Regular Neuron Networs: 

Input --> hidden layers (neurons are fully connected to prev.Layer) --> output (FC) 

 

E.g. <32*32*3> image = 3072 weights in 1. hidden layer.Wasteful, too many parameters lead to overfitting. 

 

In CNN: explicit IMAGES! Hence: Neurons arearranged in 3 dim W*H*D.

![depth height width ](file:///C:/Users/hasee/AppData/Local/Packages/Microsoft.Office.OneNote_8wekyb3d8bbwe/TempState/msohtmlclip/clip_image001.png)

 

Main types of layers: 

- **Input**: hold the raw pixel     velues of the image. E.g. 32*32*3 color channels.
- **Conv Layer**: each neurons     will compute dot product between their weights and the region they are     connected to. Per Layer $K*(F*F*D+1)$ Parameters when using Parameter     sharing.
- **ReLu**: elementwise     activation function, such as max(0,x) thersholding at zero. Size is     unchanged. (fixed func, no parameters)
- **Pooling**: downsampling     operation along the spatial dimensions. (fixed func, no parameters)
- **Fully-Connected Layer**:     compute the class scores. Parameters in Conv/FC Layers will be trained     with gradient descent. 



## Convolutional Layers: 

 

### Convolutional Layer

 

**Local connectivity**: each neuron would be connect to a local region of the input volume. The Spatial extent is a hyperparameter called the receptive field of the neuron(F); Depth is the same as the depth of the input volume. Thus Connections are local in space, but full along the entire depth. 

 

Spatial arrangement: 

- Depth: a set of neurons     that are looking at the same region = depth column / fibre.
- Stride
- Zero-padding: nice to     control the spatial size of the output. 
- Output would be of size $(W-F+2P)/S+1$. Noted that the size should always be an integer. Otherwise     the neurons don't "fit".      Easy to see that when P=(F-1)/2, S=1, the spatial size of the input     volume will be preserved.  



**Parameter Sharing**:  with in adepth slice/activation map. W*H neurons in the same depth slice(say K=1, thesecond filter) will share the same weights und bias. Hence the nameConvolutional Layer. Sometimes it may not make sense, especially in the casewhen the input have some specific centered structure, which lead us tolocally-connected Layer.

 

### Numpy examples>

Input volume = numpy array X.

Depth column(fibre) at position(x,y) = activations X[x,y,:]

Depth slice/activation map at death d = activations X[:,:,d]

 

### Conv Layer Example 

Input volume> X.shape:(11,11,4),

No zero padding (P=0), Filter size F=5, stride S=2.

Output volume> (W-F)/S+1=(11-5)/2+1=4, 4x4 as spatial size.

Activation map V>  first 4

![code](file:///C:/Users/hasee/AppData/Local/Packages/Microsoft.Office.OneNote_8wekyb3d8bbwe/TempState/msohtmlclip/clip_image002.png)

W0.shape:(5,5,4) (size of filter, depth of the input volume)

At each point: dot product. Same Weight und bias in the same depthslice d=0. 

 

The second depth slice (overall there would be K slices, if Kfilters would be used in this conv. Layer): different set of parameters W1 isused.

![v CO, o, VCI, O, VC2, o, VC3, o, vco, 1 VC2, 3, np. sum (XC: 5, np. sum (XC4•9 np. :5, * WI) + bl np. 2: 7, : ) * WI) + bl (example of going along y) np. * WI) + bf (or along both) ](file:///C:/Users/hasee/AppData/Local/Packages/Microsoft.Office.OneNote_8wekyb3d8bbwe/TempState/msohtmlclip/clip_image003.png)

 

### Summary> the Conv.Layer:

- Accepts a volume of size     W1 * H1 * D1

- Requires four     hyperparameters: 

- - Number of filters K
  - Spatial extend F
  - Stride S
  - Amount of zero padding P

- Produces a volume of size     W2 * H2 *D2

- - W2=(W1-F+2P)/S+1
  - H2=(H1-F+2P)/S+1 computed      equally by symmetry
  - D2=K

- With parameter sharing, it     produces F*F*D1 weights per filter. In total: K*F*F*D1 weights, K biases. 

- In the output volume, the d-th depth slice(W2*H2) is     the result of performing a valid convolution of the d-th filter over the input     with Stride S, and then offset by d-th bias.

- A common setting of the     hyper parameters> F=3, S=1, P=1. 



###  Implementation as MatrixMultiplication>

1. The local regions in the input images are stretched out into     columns in an operation commonly call im2col. Input     227*227*3, convolved with 11*11*3 filters at stride 4, produces a 55*55     slice>> every column is a streched out receptive field with 11*11*3     = 363 elements, and there are 55*55 = 3,025 columns . Output matrix X_col:     363 * 3025. 
2. The weights of the CONV     layer are similarly stretched out into rows. E.g. 96 filters of size     11*11*3> Matrix W_row of size 96*363
3. The result of a     convolution is np.dot(W_row, X_col). Dot product between every filter and     every receptive field location. Output would be 96*3025. 
4. The result must finally be     reshaped back to its proper output dimension 55*55*96. 
5. It can use a lot of     memory, since some values are replicated multiple times in X_col.     Benefit> efficient implementations that can take advantage of e.g. BLAS     API. Moreover, im2col can be reused to perform the pooling operation. 



###  Backpropagation 

the backward pass for a conv.Operation is also a convolution.

 

###  1*1 convolution. 

 For image it would effectively be doing 3-dimdot products.

 

###  Dilated convolutions: 

onemore hyperparameter called dilation. Aka Filters are no longer contiguous,  have spaces between each cell, calleddilation.  In one Dimension with a filtersize of 3> 

Dilation = 0:  

![img](file:///C:/Users/hasee/AppData/Local/Packages/Microsoft.Office.OneNote_8wekyb3d8bbwe/TempState/msohtmlclip/clip_image004.png)

Dilation = 1: 

![img](file:///C:/Users/hasee/AppData/Local/Packages/Microsoft.Office.OneNote_8wekyb3d8bbwe/TempState/msohtmlclip/clip_image005.png)

By using dilated convolutions the effective receptive field wouldgrow much quicker.

 

## Pooling Layer

 

**Function** progressively reduce the spatial size of therepresentation to reduce the amount of parameters and computation in thenetwork, and hence to also control overfitting. Operates independently on every depth slice of the input and resizes itspatially, using MAX operation (MaxPooling). Most common> a pooling layer(filters 2*2, stride 2) downsamples every depth slice in the input by 2 along Wand H, discarding 75% of the activations. Every Max operation would be nowtaking a max over 4 nums. Depth remains unchanged. 

![Accepts a volume of size WI x HI x DI Requires two hyperparameters: o their spatial extent F o the stride S Produces a volume of size VV2 x 1-12 x where. = (WI - F)/S+I = (HI - F)/S+I • Introduces zero parameters since it computes a fixed function of the input Note that it is not common to use zero-padding for Pooling layers ](file:///C:/Users/hasee/AppData/Local/Packages/Microsoft.Office.OneNote_8wekyb3d8bbwe/TempState/msohtmlclip/clip_image006.png)

Two commonly seen variations of the max pooling layer: overlapping pooling with F=3, S=2. or more commonly F=2, S=2.

 

### General pooling 

max, average, L2-norm pooling

### Backpropagation

 during the forward passof a pooling layer it is common to keep track of the index of the maxactivation, so that gradient routing is efficient during backpropagation. 

### Getting rid of pooling

using larger stride oncein a while etx.  Also important intraining good generative models such as VAEs, GANs.

 

 

## Normalization Layer 

 

## Fully-Connected Layer

Neuronsin this layer have full connections to all activations in the prev. layers.Their activations can hence be computed with a matrix multiplication followedby a bias offset.  To get the classscores for example.

 

## Converting FC to Conv. 

 

# ConvNet Architectures

 

# Layer patterns 

Most common> stacks a few CONV-RELU layers, follows them withPOOL Layers, repeats until the image has been merged spatially to a small size.Transition to fully-connected layers. The last FC layer hold the output such asthe class scores. 

![img](file:///C:/Users/hasee/AppData/Local/Packages/Microsoft.Office.OneNote_8wekyb3d8bbwe/TempState/msohtmlclip/clip_image007.png)

 

![img](file:///C:/Users/hasee/AppData/Local/Packages/Microsoft.Office.OneNote_8wekyb3d8bbwe/TempState/msohtmlclip/clip_image008.png)

![INPUT -> FC , implements a linear classifier. INPUT -> COW -> RELU -> FC INPUT -> CCOW -> RELU -> -> FC -> RELU CONV layer between every POO layer. FC Here we see that there is a single INPUT -> CCOW -> RELU -> COW -> RELU -> -> CFC -> -> pc I Here we see two CONV layers stacked before every POOL layer. This is generally a good idea for larger and deeper networks, because multiple stacked CONV layers can develop more complex features of the input volume before the destructive pooling operation. ](file:///C:/Users/hasee/AppData/Local/Packages/Microsoft.Office.OneNote_8wekyb3d8bbwe/TempState/msohtmlclip/clip_image009.png)

 

Intuitively,stacking CONV layers with tiny filters as opposed to having one CONV layer withbig filters allows us to express more powerful features of the input, and withfewer parameters. As a practical disadvantage, we might need more memory tohold all the intermediate CONV layer results if we plan to do backpropagation.

 

Insteadof rolling your own architecture for a problem, you should look at whateverarchitecture currently works best on ImageNet, download a pretrained model andfinetune it on your

 

 

## Layer Sizing Patterns

 

Input layer> should be divisible by 2 many times. Aka 2^x. 32 ,64, 96, 224, 384, 512

 

Conv layers> using small filters 3, 5. a stride of S=1, paddingwith zeros in such way that conv layer does not alter spatial dimensions of theinput. F=3, P=1; F=5, P=2; P= (F-1)/2! If muss be use bigger F, then onlycommon on very first conv layer.

 

Pool Layer> in charge of downsampling the spatial dim. F2, S2.discard exactly 75% of the activations in an input volume. Less common F3, S2.overlapping pooling. Very uncommon when larger than 3, too lossy andaggressive. 

 

Why S=1> smaller strides work better and allows us to leave allspatial down-sampling to the POOL layers. 

 

Why padding> keep spatial sizes constant after Conv. Otherwisethe size reduce too fast, and the information at the borders would be washedaway too quickly

 

Compromising based on memory constraints.

 

#  Case Studies

- LeNet
- AlexNet
- ZF Net
- GoogleLeNet using inception
- VGGNet
- ResNet



Thewhole VGGNet is composed of CONV layers that perform 3x3 convolutions withstride 1 and pad 1, and of POOL layers that perform 2x2 max pooling with stride2 (and no padding).

 

Most of the memory is used in the early CONV Layers. Most of theparameters are in the last FC Layers. 

 

 