---
title: Neural Network Quantization	
categories:
- Quantization
feature_text: 
  Neural Network Quantization
feature_image: "https://picsum.photos/2560/600?image=872"
excerpt: "Process of reducing the number of bits used to represent integer is termed as quantization. In this blog we will look into different types of quantization and quantization techniques used by tensorflow lite."
---

<script type="text/javascript" src="http://latex.codecogs.com/latexit.js"></script>

<h4>Quantization</h4>
<div>
	<p>
	 Process of reducing the number of bits used to represent integer is termed as quantization. Numerical format used in deep learning libraries is 32 bit floating point representation. We will see about it in the next section. But applications of deep learning models in the edge devices has raised the need for the lower precision numerical formats like int. There is a good amount of literature which shows that weights and activations of the neural network can be repsented with 8 bit integers without much loss in accuracy. 	
	</p>
</div>

<h5>Fixed Point vs Floating point Representation</h5>

<div>	
<h6>Fixed point Representation</h6>
<p>
	32 bits are divided into two parts. First part contains the integer part of the number and the second part contains the fractional part of the number.<br>
</p>

<p style="margin-bottom:0.1cm;"> </p>
<table align="center">
  <tr><th align="center">Example</th><th align="center">Format</th></tr>
  <tr><td>Number</td><td>(integer part).(fractional part)</td></tr>
  <tr><td>Largest number</td><td>111....1    . 111....1</td></tr>
  <tr><td>Smallest number</td><td>000....0    . 0....1</td></tr>
</table>
<p style="margin-top:0.1cm;"> </p>

<h6>Problem</h6>
<p>
	Range represented by fixed point representation is not suitable for practical problems. Hence floating point representation is used.
</p>
<h6>Floating point Representation</h6>
<p>
	32 bits are divided into two parts. First part contains matissa with its sign and the second part contains teh exponent with its sign. Any number using floating point representation can be written as 
	<br><p style="margin:0.01cm;"></p>(sign) * (mantissa) * 2 <sup>exponent</sup><br>
</p>


Since there is no fixed bits representation for integer part and fractional part this rises following questions
<ol>
	<li>How many bits to represent mantissa?</li>
	<li>How many bits to represent exponent?</li>
	<li>How to represent sign of exponent?</li>
</ol>

<ol>
	<li>Based on experience atleast 7 digits are needed if results are expected without much error. Number of bits to represent 7 digits is 23. This requires 23 bits for mantissa and 1 bit for sign of mantissa</li>
	<li>Remaning 8 bits are allocated for the exponent. 1 for sign and other 7 for exponent.</li>
</ol>
</div>


<h5>Conservative Quantization</h5>
In general taking a FP32 model and quantised for int8 results in a low loss of accuracy. Some finetuning can increase the accuracy of quantized model. When used to directly quantize a model without re-training, as described so far, this method is commonly referred to as post-training quantization. Smaller models such as MobileNet seem to not respond as well to post-training quantization, presumabley due to their smaller representational capacity.


<h5>Aggressive Quantization</h5>
Considering FP32 model and quantizing it to int4 or lower representation results in significant loss of accuracy. Most commonly used methods for quantizing to int4 are retraining, cliiping values or changing relu activation, modifying network structure, not quantizing first and last layers of the network. It has been observed that activations are more sensitive to quantization than weights.


<h4>Quantization method in TfLite</h4>

<!-- <h5>Quantized Inference</h5> -->
<h6>Quantization Scheme</h6>
<p>
	This quantization scheme is implemented using only integer arithmetic in inference and floating point arithmetic during training. Quantization scheme be an thought of as an affine mapping of integers q (quantized value) to real numbers r. Quantization scheme can be written as <span lang="latex"> r = S(q-Z) </span> where S and Z are the quantization parameters.
</p>

<p>This quantization scheme uses different set of parameters for each activation array. Separate arrays use separate quantization parameters. In general bias are quantised as 32 bit vectors. Constant S is an positive real number. Constant Z (zero point) is same type as of quantized values. Real value 0 is exactly represented by quantized value. The motivation
for this requirement is that efficient implementation of neural network operators often requires zero-padding of arrays around boundaries.</p>


<h5>Training with simulated quantization</h5>

<p>
	General way to obtain a quantized model is to train a network with FP32 representation and quantize the resulting weights. This works well for the larger networks but fails miserably for the smaller models like Mobilenets. Common causes for such significant drop in accuracy are large differences in weights and outlier weight values. More than 100X difference in range of weights can cause weights in channels with smaller changes to have relatively high errror. Outlier weight values can make all remaning weights less precise.
</p>

<p>To overcome these problems, quantization aware training is used. This simulates the quantization effects in the forward pass of the network and backpropagation happens as usual. Weights and bias are initialised in floating point so that they can be trained easily. </p>

<p>
	Weights are convolved before the convolution and if batch norm is used batchnorm is folded into weights. Activations are quantized after activation function is applied to convolution layer or fully connected layer output or after add operation in resnets. 
</p>

<h6>Learning Quantization Ranges</h6>
<p>
	Quantization ranges are treated differently for weight quantization and activation quantization. For weights idea is to set a =min(w) and b = max(w) where a and b form the quantization range. For activation range depends on the inputs so we take the [a,b] ranges for activations during training and then use exponential moving average to calculate the final quantization ranges. 
</p>
<h6>Batch Normalization folding</h6>
<p>
	For models that use batch normalization, there is additional complexity: the training graph contains batch normalization as a separate block of operations, whereas the inference graph has batch normalization parameters folded into the convolutional or fully connected layer’s weights and biases, for efficiency. To accurately simulate quantization effects, we need to simulate this folding.
</p>

<p>
	After folding, the batch-normalized convolutional layer reduces to the simple convolutional layer with the folded weights w<sub>fold</sub>  and the corresponding folded biases.
</p>