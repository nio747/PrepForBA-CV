# Layer in ConvNet:


## Convolutional Layers: 

### Numpy examples
		Input volume = numpy array X.
		Depth column(fibre) at position(x,y) = activations X[x,y,:]
		Depth slice/activation map at death d = activations X[:,:,d]
		
		Conv Layer Example> 
			Input volume> X.shape:(11,11,4),
			No zero padding (P=0), Filter size F=5, stride S=2.
			Output volume> (W-F)/S+1=(11-5)/2+1=4, 4x4 as spatial size.
			Activation map V>  first 4
			
				W0.shape:(5,5,4) (size of filter, depth of the input volume)
				At each point: dot product. Same Weight und bias in the same depth slice d=0. 
				
			The second depth slice (overall there would be K slices, if K filters would be used in this conv. Layer): different set of parameters W1 is used.
				
				
### Summary> the Conv. Layer:
		○ Accepts a volume of size W1 * H1 * D1
		○ Requires four hyperparameters: 
			§ Number of filters K
			§ Spatial extend F
			§ Stride S
			§ Amount of zero padding P
		○ Produces a volume of size W2 * H2 *D2
			§ W2=(W1-F+2P)/S+1
			§ H2=(H1-F+2P)/S+1 computed equally by symmetry
			§ D2=K
		○ With parameter sharing, it produces F*F*D1 weights per filter. In total: K*F*F*D1 weights, K biases. 
		○ In the output volume, the d-th depth slice(W2*H2) is the result of performing a valid convolution of the d-th filter over the input with Stride S, and then offset by d-th bias.
		○ A common setting of the hyper parameters> F=3, S=1, P=1. 
	
	###Implementation as Matrix Multiplication>
		1. The local regions in the input images are stretched out into columns in an operation commonly call im2col. Input 227*227*3, convolved with 11*11*3 filters at stride 4, produces a 55*55 slice>> every column is a streched out receptive field with 11*11*3 = 363 elements, and there are 55*55 = 3,025 columns . Output matrix X_col: 363 * 3025. 
		2. The weights of the CONV layer are similarly stretched out into rows. E.g. 96 filters of size 11*11*3> Matrix W_row of size 96*363
		3. The result of a convolution is np.dot(W_row, X_col). Dot product between every filter and every receptive field location. Output would be 96*3025. 
		4. The result must finally be reshaped back to its proper output dimension 55*55*96. 
		5. It can use a lot of memory, since some values are replicated multiple times in X_col. Benefit> efficient implementations that can take advantage of e.g. BLAS API. Moreover, im2col can be reused to perform the pooling operation. 
	
	### Backpropagation>  the backward pass for a conv. Operation is also a convolution.
	
	### 1*1 convolution.  For image it would effectively be doing 3-dim dot products.
	
	### Dilated convolutions : one more hyperparameter called dilation. Aka Filters are no longer contiguous,  have spaces between each cell, called dilation.  In one Dimension with a filter size of 3> 
	Dilation = 0:  
	
	Dilation = 1: 
	
	By using dilated convolutions the effective receptive field would grow much quicker.
	
## Pooling Layer

Function> progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overfitting.  Operates independently on every depth slice of the input and resizes it spatially, using MAX operation (MaxPooling). Most common> a pooling layer (filters 2*2, stride 2) downsamples every depth slice in the input by 2 along W and H, discarding 75% of the activations. Every Max operation would be now taking a max over 4 nums. Depth remains unchanged. 
	
Two commonly seen variations of the max pooling layer: overlapping pooling with F=3, S=2. or more commonly F=2, S=2.

### General pooling> max, average, L2-norm pooling
### Backpropagation> during the forward pass of a pooling layer it is common to keep track of the index of the max activation, so that gradient routing is efficient during backpropagation. 
### Getting rid of pooling , using larger stride once in a while etx.  Also important in training good generative models such as VAEs, GANs.


## Normalization Layer 

## Fully-Connected Layer
Neurons in this layer have full connections to all activations in the prev. layers. Their activations can hence be computed with a matrix multiplication followed by a bias offset.  To get the class scores for example.

## Converting FC to Conv. 
	
# ConvNet Architectures

	Layer patterns 
		Most common> stacks a few CONV-RELU layers, follows them with POOL Layers, repeats until the image has been merged spatially to a small size. Transition to fully-connected layers. The last FC layer hold the output such as the class scores. 
		
		
		
		
		
		Intuitively, stacking CONV layers with tiny filters as opposed to having one CONV layer with big filters allows us to express more powerful features of the input, and with fewer parameters. As a practical disadvantage, we might need more memory to hold all the intermediate CONV layer results if we plan to do backpropagation.
		
		Instead of rolling your own architecture for a problem, you should look at whatever architecture currently works best on ImageNet, download a pretrained model and finetune it on your
		
		
	Layer Sizing Patterns
		
		Input layer> should be divisible by 2 many times. Aka 2^x. 32 , 64, 96, 224, 384, 512
		
		Conv layers> using small filters 3, 5. a stride of S=1, padding with zeros in such way that conv layer does not alter spatial dimensions of the input. F=3, P=1; F=5, P=2; P= (F-1)/2! If muss be use bigger F, then only common on very first conv layer.
		
		Pool Layer> in charge of downsampling the spatial dim. F2, S2. discard exactly 75% of the activations in an input volume. Less common F3, S2. overlapping pooling. Very uncommon when larger than 3, too lossy and aggressive. 
		
		Why S=1> smaller strides work better and allows us to leave all spatial down-sampling to the POOL layers. 
		
		Why padding> keep spatial sizes constant after Conv. Otherwise the size reduce too fast, and the information at the borders would be washed away too quickly
		
		Compromising based on memory constraints.
		
	Case Studies
		LeNet
		AlexNet
		ZF Net
		GoogleLeNet using inception
		VGGNet
		ResNet
		
		The whole VGGNet is composed of CONV layers that perform 3x3 convolutions with stride 1 and pad 1, and of POOL layers that perform 2x2 max pooling with stride 2 (and no padding).
		
		Most of the memory is used in the early CONV Layers. Most of the parameters are in the last FC Layers. 
		
		
# Layer in ConvNet:


## Convolutional Layers: 

	### Numpy examples>
		Input volume = numpy array X.
		Depth column(fibre) at position(x,y) = activations X[x,y,:]
		Depth slice/activation map at death d = activations X[:,:,d]
		
		Conv Layer Example> 
			Input volume> X.shape:(11,11,4),
			No zero padding (P=0), Filter size F=5, stride S=2.
			Output volume> (W-F)/S+1=(11-5)/2+1=4, 4x4 as spatial size.
			Activation map V>  first 4
			
				W0.shape:(5,5,4) (size of filter, depth of the input volume)
				At each point: dot product. Same Weight und bias in the same depth slice d=0. 
				
			The second depth slice (overall there would be K slices, if K filters would be used in this conv. Layer): different set of parameters W1 is used.
				
				
	### Summary> the Conv. Layer:
		○ Accepts a volume of size W1 * H1 * D1
		○ Requires four hyperparameters: 
			§ Number of filters K
			§ Spatial extend F
			§ Stride S
			§ Amount of zero padding P
		○ Produces a volume of size W2 * H2 *D2
			§ W2=(W1-F+2P)/S+1
			§ H2=(H1-F+2P)/S+1 computed equally by symmetry
			§ D2=K
		○ With parameter sharing, it produces F*F*D1 weights per filter. In total: K*F*F*D1 weights, K biases. 
		○ In the output volume, the d-th depth slice(W2*H2) is the result of performing a valid convolution of the d-th filter over the input with Stride S, and then offset by d-th bias.
		○ A common setting of the hyper parameters> F=3, S=1, P=1. 
	
	### Implementation as Matrix Multiplication>
		1. The local regions in the input images are stretched out into columns in an operation commonly call im2col. Input 227*227*3, convolved with 11*11*3 filters at stride 4, produces a 55*55 slice>> every column is a streched out receptive field with 11*11*3 = 363 elements, and there are 55*55 = 3,025 columns . Output matrix X_col: 363 * 3025. 
		2. The weights of the CONV layer are similarly stretched out into rows. E.g. 96 filters of size 11*11*3> Matrix W_row of size 96*363
		3. The result of a convolution is np.dot(W_row, X_col). Dot product between every filter and every receptive field location. Output would be 96*3025. 
		4. The result must finally be reshaped back to its proper output dimension 55*55*96. 
		5. It can use a lot of memory, since some values are replicated multiple times in X_col. Benefit> efficient implementations that can take advantage of e.g. BLAS API. Moreover, im2col can be reused to perform the pooling operation. 
	
	### Backpropagation>  the backward pass for a conv. Operation is also a convolution.
	
	### 1*1 convolution.  For image it would effectively be doing 3-dim dot products.
	
	### Dilated convolutions : one more hyperparameter called dilation. Aka Filters are no longer contiguous,  have spaces between each cell, called dilation.  In one Dimension with a filter size of 3> 
	Dilation = 0:  
	
	Dilation = 1: 
	
	By using dilated convolutions the effective receptive field would grow much quicker.
	
## Pooling Layer

Function> progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overfitting.  Operates independently on every depth slice of the input and resizes it spatially, using MAX operation (MaxPooling). Most common> a pooling layer (filters 2*2, stride 2) downsamples every depth slice in the input by 2 along W and H, discarding 75% of the activations. Every Max operation would be now taking a max over 4 nums. Depth remains unchanged. 
	
Two commonly seen variations of the max pooling layer: overlapping pooling with F=3, S=2. or more commonly F=2, S=2.

### General pooling> max, average, L2-norm pooling
### Backpropagation> during the forward pass of a pooling layer it is common to keep track of the index of the max activation, so that gradient routing is efficient during backpropagation. 
### Getting rid of pooling , using larger stride once in a while etx.  Also important in training good generative models such as VAEs, GANs.


## Normalization Layer 

## Fully-Connected Layer
Neurons in this layer have full connections to all activations in the prev. layers. Their activations can hence be computed with a matrix multiplication followed by a bias offset.  To get the class scores for example.

## Converting FC to Conv. 
	
# ConvNet Architectures

	Layer patterns 
		Most common> stacks a few CONV-RELU layers, follows them with POOL Layers, repeats until the image has been merged spatially to a small size. Transition to fully-connected layers. The last FC layer hold the output such as the class scores. 
		
		
		
		
		
		Intuitively, stacking CONV layers with tiny filters as opposed to having one CONV layer with big filters allows us to express more powerful features of the input, and with fewer parameters. As a practical disadvantage, we might need more memory to hold all the intermediate CONV layer results if we plan to do backpropagation.
		
		Instead of rolling your own architecture for a problem, you should look at whatever architecture currently works best on ImageNet, download a pretrained model and finetune it on your
		
		
	Layer Sizing Patterns
		
		Input layer> should be divisible by 2 many times. Aka 2^x. 32 , 64, 96, 224, 384, 512
		
		Conv layers> using small filters 3, 5. a stride of S=1, padding with zeros in such way that conv layer does not alter spatial dimensions of the input. F=3, P=1; F=5, P=2; P= (F-1)/2! If muss be use bigger F, then only common on very first conv layer.
		
		Pool Layer> in charge of downsampling the spatial dim. F2, S2. discard exactly 75% of the activations in an input volume. Less common F3, S2. overlapping pooling. Very uncommon when larger than 3, too lossy and aggressive. 
		
		Why S=1> smaller strides work better and allows us to leave all spatial down-sampling to the POOL layers. 
		
		Why padding> keep spatial sizes constant after Conv. Otherwise the size reduce too fast, and the information at the borders would be washed away too quickly
		
		Compromising based on memory constraints.
		
	Case Studies
		LeNet
		AlexNet
		ZF Net
		GoogleLeNet using inception
		VGGNet
		ResNet
		
		The whole VGGNet is composed of CONV layers that perform 3x3 convolutions with stride 1 and pad 1, and of POOL layers that perform 2x2 max pooling with stride 2 (and no padding).
		
		Most of the memory is used in the early CONV Layers. Most of the parameters are in the last FC Layers. 
		
		
