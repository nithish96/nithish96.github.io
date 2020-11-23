---
title: MobileNets
categories:
- Computer Vision
feature_text: |
  Mobilenets : Efficient Convolutional Neural Networks
feature_image:
   "https://picsum.photos/2560/600?image=872"
excerpt: "In this post we will look at the recent advancements of neural networks for edge devices. Specifically we will look at family of architectures called mobilenets."

---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<body>

<div>
	<h3>Introduction</h3>
	<p>We see a lot of neural networks these days, that can achieve the human level performance in many tasks. In general neural networks are trained using high performance computing machines that use GPU. Deployment of traditional neural networks on edge devices is slightly complex because of computational complexity and memory constraints. Given these constraints there is an emerging need for developers to build models that can produce accurate results in real time. So today we will look at class of efficient networks called Mobilenets that help us in developing simpler networks that perform similar to state of the art networks. </p>
	<h3>MobileNet</h3>
		<p>Mobilenet architecture is primarily based on depthwise convolutions and pointwise convolutions. We will discuss about these in the following section. </p>
		<h4>Depthwise Convolutions</h4>
			<p>
				Depthwise convolutions are nothing but the factorized form of convolutions. Convolution operation is factorized into two - depthwise convolution and pointwise convolution. Depthwise convolution applies a single filter to each input channel and pointwise convolution performs 1 X 1 convolution to give layer output. Standard convolution filters and combines in one step where as depthwise convolution does in two steps - apply filters and then combine. This factorization reduces lot of computation. 
			</p>
			<p>To make things clear, let us assume an RGB image of size \( (100, 100, 3) \) and 64 filters of size \( (5, 5, 3) \) are applied on input. Computational complexity of a standard convolution can be written as (100 * 100 * 3* 5 * 5 * 64). This computation can formally  be written as  \( (x_i * x_j * c_{in} *  k_i * k_j * N) \) where x is the input with shape \( (x_i, x_j, c_{in})\)and k is the kernel of size \( (k_i, k_j) \) and N is the number of filters. Standard convolution has a effect of filtering features and then combining features to produce feature representation. </p>
			<p>Depthwise convolution apply a single filter per each input channel. Pointwise convolution then creates linear combination of output of depthwise layer. Computational cost of depthwise convolution for the above example would be (100 * 100 * 3 * 5 * 5). 	Computational cost of pointwise convolution is (100 * 100 * 3 * 64).  Total cost is defined as cost of depthwise convolution + cost of pointwise convolution i.e  \( ( x_i * x_j * c_{in}* k_i * k_j ) \) + \( (x_i * x_j * c_{in} * N)\). So we get a computation reduction of 8 - 9 times with depthwise convolutions for a kernel of \( (3, 3) \). </p>
			<center>
			 <figure>
			  <img src="/assets/mobilenet_v1_block.png" style="width:550px;height:400px;" >
			  <figcaption>Fig 1 - Standard Convolution vs Depthwise convolution </figcaption>
			</figure>
			</center>
		<h4>Network Structure</h4>
			<p>Mobilenet consists of depthwise convolution except the first layer which is a standard convolution. All layers are followed by batch norm and relu except the last fully connected layer which connects to softmax layer. Downsampling is done with strided convolution instead of pooling. Max pooling makes model translation invariant where as strided convolution makes network faster, simpler and downsampling layer specific (some layers may need higher downsampling than others). Mobilenet has a depth of 28 layers. </p>
			<p>Mobilenet spends 95% of the time in pointwise convolution which has 75% of the parameters. 24% of the parameters are attributed to fully connected layer. We use less regularization and data augmentation because smaller models are less prone to overfitting. It was found that small amount of l2 regularization was needed for depthwise filters. </p>
			<p>Mobilenet has a parameter called width multiplier \( \alpha\). \( \alpha\) = 1 represents the base mobilenet and \( \alpha\) < 1 represent the reduced mobilenets. Width multiplier can be applied to any structure to define a new smaller model with reasonable accuracy, latency and size tradeoff. Width multiplier can reduce computational cost and number of parameters by roughly \( \alpha^2\). </p>
			<p>Mobilenet has a second parameter called resolution multiplier \( \rho\). This is used to reduce computational cost of network. We apply this to input layer and internal representaiton of every layer is reduce by same multiplier. \( \rho\) = 1 is baseline mobilenet and \( \rho\) < 1 are reduced mobilenets. Width multiplier is used tor reduce the number of parameters in the network where as resolution multiplier is used to reduce number of flops or multiadds in the network.
			</p>
	<h3>MobilenetV2</h3>
		<p>Mobilenetv2 models are much faster than mobilenet v1. It uses 2 times fewer operations with 30% fewer parameters and has higher accuracy then mobilenet v1. Mobilenetv2 has 3 convolutional layers in each block - expansion layers, depthwise convolution, projection layer. Expansion layer works in the same way as pointwise convolution in mobilenet v1 - it expands the number of channels by a expansion factor. Default expansion factor used in the paper is 6.   </p>
		<center>
			 <figure>
			  <img src="/assets/mobilenet_v2_block.png" style="width:300px;height:350px;" >
			  <figcaption>Fig 1 - MobileNetv2 block. Image from [3].</figcaption>
			</figure>
		</center>
		<p>Generally channels are increased and  spatial dimensions are reduced by half as we go deep into the network. Using low dimensional tensors is the key to reduce the number of computations. Downside of this is applying convolution on a low dimensional tensor may not give us the useful information. Mobilenet gives best of both worlds. </p>
		<p>Expansion layer acts as a decompressor on the input data by restoring the data to its original form then depthwise does feature learning on the original data and at last projection layer compresses the data to feed it to the next layer. Since all the three layers i.e depthwise, expansion, projection are done using learnable parameters the model learns how to decompress and compress the data. </p>
		<p>Constant expansion factor is used through out the network. Smaller network tend to perform with smaller expansion factors and vice verca. Mobilenetv2 primary network has a computational cost of 300 Million flops and uses 3.4 Million parameters. Network computational cost ranges from 7 Million flops to 585 Million flops while the model size can vary form 1.7 MB and 7MB. Applying width multiplier to all the layers except the last convolution layer improves the performance for smaller layers.  </p>
	<h3>Results</h3>
		<p>Having seen the architectural details of these models the question that arise is where do they stand. Results of mobilenet architecture on imagenet are as follows </p>
		<center>
			 <figure>
			  <img src="/assets/v1_vs_v2.png" style="width:614px;height:420px;" >
			  <figcaption>Fig 1 - MobileNetv1 vs MobileNetv2. Image from [4].</figcaption>
			</figure>
		</center>
		<p>In all the metrics Mobilenetv2 has produced better results than v1. Mobilenet can act as efficient baseline for many visual recognition tasks like object detection or image segmentation. </p>
	<h3>References</h3>
		<ol>
			<li><a href="https://arxiv.org/pdf/1704.04861.pdf">Mobilenet v1</a></li>
			<li><a href="https://arxiv.org/pdf/1801.04381.pdf">Mobilenet v2</a></li>
			<li><a href="https://machinethink.net/blog/mobilenet-v2/" >Mobilenet version 2 </a></li>
			<li><a href="https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html" >MobileNetV2: The Next Generation of On-Device Computer Vision Networks</a></li>
		</ol>
</div>
</body>