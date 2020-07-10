---
title: Neural Network Quantization
categories:
- Quantization
feature_text:
  Neural Network Quantization
feature_image: "https://picsum.photos/2560/600?image=872"
excerpt: "Process of reducing the number of bits used to represent integer is termed as quantization. In this blog we will look into different types of quantization and quantization techniques used by tensorflow lite."
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<body>
<h2>Introduction</h2>
<div>
	<p>
	 Process of reducing the number of bits used to represent integer is termed as quantization. Numerical format used in deep learning libraries is 32 bit floating point representation. Applications of deep learning models in the edge devices has raised the need for the lower precision numerical formats like int8. There is a good amount of literature which shows that weights and activations of the neural network can be represented with 8 bit integers without much loss in accuracy. 	
	</p>
  <p>
    Before diving into quantization lets go through some numerical representations. We have only certain bits to represent a real number in a computer. Some of the important ways to represent a number are fixed point representation and floating point representation. 
  </p>
</div>
    
<h2>Fixed Point vs Floating point Representation</h2>

<div>
<h3>Fixed point Representation</h3>
<p>
	32 bits are divided into two parts. First part contains the integer part of the number and the second part contains the fractional part of the number.<br>
</p>

<table>
  <tr>
    <th></th>
    <th>Largest</th>
    <th>Smallest</th>
  </tr>
  <tr>
    <td>Written as </td>
    <td>(111....1).(111....1)</td>
    <td>(000....0).(0....1)</td>
  </tr>
  <tr>
    <td>Value</td>
    <td>1677215.9980</td>
    <td>0.00390</td>
  </tr>
</table>
<br>
<p>
	Range represented by fixed point representation is not suitable for practical problems. Hence floating point representation is used to represent real numbers in computers.
</p>
<h3>Floating point Representation</h3>
<p>
	32 bits are divided into two parts. First part contains mantissa with its sign and the second part contains the exponent with its sign. Any number using floating point representation can be written as	<br><b>(sign) * (mantissa) * 2 <sup>exponent</sup></b><br>
</p>


mantissa represents the fractional part and 2 power exponent is multiplied with exponent to get real number. Since there is no fixed bits representation for integer part and fractional part this rises following questions
<ol>
	<li>How many bits to represent mantissa?</li>
	<li>How many bits to represent exponent?</li>
	<li>How to represent sign of exponent?</li>
</ol>

<ol>
	<li>Based on experience atleast 7 significant decimal digits are needed if results are expected without much error. Number of bits to represent 7 digits is 23. This requires 23 bits for mantissa and 1 bit for sign of mantissa</li>
	<li>Remaning 8 bits are allocated for the exponent. 1 for sign and other 7 for exponent. So exponent will range from -127 to +127. Exponent written this way 0 will have two representations of 0 i.e +0 and -0. Hence range of 0 to 255 is used.</li>
</ol>
<table>
  <tr>
    <th></th>
    <th>Largest</th>
    <th>Smallest</th>
  </tr>
  <tr>
    <td>Written as </td>
    <td>0.11 ... 11 × 2<sup>(255-127)</sup> </td>
    <td>0.10 ... 00 x 2<sup>-127</sup></td>
  </tr>
  <tr>
    <td>Value</td>
    <td>~ 3.4 × 10 <sup>38</sup></td>
    <td> ~ 0.293 x 10 <sup>-38</sup></td>
  </tr>
</table>
<p></p>
</div>


<h2>Why does quantization work?</h2>
<p>
 There is no mathematical reason behind why quantization works. It emerges as a side effect of successful training. When we learn a network we expect it to perform reasonably well with outliers or the adverse data. Dropout is one such example where network can work well with adverse data. Network that emerge from training have to be robust numerically so that small differences in input would not affect the output. Compared to variations in pose, position and light noise in the image is a simple problem. All layers in a network will develop tolerance to changes in input. That means the differnces obtained by the low precision are with in those tolerances that network has learnt. </p>
 <p>
  Training a neural network is done by updating weights and these updates need floating point precision to work. Quantized networks seem to treat low precision values as just another source of noise and still produce accurate results with numerical formats that hold less information.
</p>
<h2>Why Quantization?</h2>
  Quantization shrinks memory by storing min and max for each layer in the network and then compressing each float value to lower precision like int8. Once model is loaded you can convert weights to floating values so that model can produce accurate results. Reading eight bit values uses 25% of bandwidth of floating point values. Quantized models run faster and use less power on devices.


<h2>Quantization and its Types</h2>

<h3>Conservative Quantization</h3>
In general taking a FP32 model and quantizing for int8 results in a low loss of accuracy. Some finetuning can increase the accuracy of quantized model. Scale factor is used to adapt the range of tensor to the integer range. Simplest value is to map min/max of float tensor to min/max of integer format. When a model is quantized without any re-training, this method is commonly referred to as post-training quantization. Smaller models such as MobileNet doesnot seem to respond to post-training quantization, presumably due to their smaller representational capacity.


<h3>Aggressive Quantization</h3>
Considering FP32 model and quantizing it to int4 or lower representation results in significant loss of accuracy. Most commonly used methods for quantizing to int4 are retraining with quantization, clipping values or changing relu activation(range is unlimited), modifying network structure, not quantizing first and last layers of the network. It has been observed that activations are more sensitive to quantization than weights.


<h2>Quantizer Designs</h2>

<h3>Uniform Affine Quantizer</h3>
  Consider a floating point range \( (x_{min}, x_{max}) \) that needs to be quantized to the range \( (0, N-1) \) where N=256 for 8 bit quantization. We compute two parameters scale \( (\Delta) \) and Zero point (z) - that map the floating point range to integer range. Scale maps step size of the quantizer and floating point zero maps to zero point. Zero point is an integer with 0 to make sure that value 0 is quantized without any error or else zero padding can cause quantization error.
  <p>
    For one sided distributions with range \( (x_{min}, x_{max}) \) zero is included in the range. For example floating point range \( (2.1, 3,5) \) is converted to range \( (0, 3.5) \) and then quantized. For extremely skewed distributions this can cause an low precision problem. 
  </p>
  <p>
    Once we have scale and zero point quantization can be done as follows. 
     $$ \begin{align*} 
             x_{int} &= round(\frac{x}{\Delta} ) + z  \\
             x_Q &= clamp(0, N_{levels}-1, x_{int})  \\
            \end{align*} $$
    where 
     $$ \begin{align*} 
        clamp(a,b,x) & = a  \hspace{0.5em} \text{if} \hspace{0.5em}x \lt a \\
                       & = b \hspace{0.5em} \text{if} \hspace{0.5em}x\gt b \\
                       &= x \hspace{0.5em} \text{if} \hspace{0.5em}a \lt x \lt b 
       \end{align*} $$
  Dequantization can be performed as 
   $$ \begin{align*} 
             x_{float} &= ( x_{Q} - z ) \Delta 
            \end{align*} $$
  </p>
<h3>Uniform symmetric quantizer</h3>
  In this quantizer we restrict the zero point to 0. With symmetric quantizer quantization can be wriiten as 
   $$ \begin{align*} 
             x_{int} &= round(\frac{x}{\Delta} )  \\
             x_Q &= clamp(\frac{-N_{levels}}{2}, \frac{N_{levels}}{2}-1, x_{int})  \\
            \end{align*} $$
    Dequantization can be performed as 
   $$ \begin{align*} 
             x_{float} &= ( x_{Q}) \Delta 
            \end{align*} $$
    In symmetric quantizer range is completely not utilized. For example operations like Relu (whose value is always +ve) we might lose some precision here.  
  
<h3>Stochastic quantizer</h3>
    Stochastic quantization models the quantization as an additive noise followed by rouding. Quantization can be written as 
  $$ \begin{align*} 
             x_{int} &= round(\frac{x + \epsilon}{\Delta}) + z  \hspace{1.5em} \epsilon \in Unif(\frac{-1}{2}, \frac{1}{2} ) \\
             x_Q &= clamp(0, N_{levels}-1, x_{int})  \\
            \end{align*} $$
    Dequantization can be performed as 
   $$ \begin{align*} 
             x_{float} &= ( x_{Q} - z ) \Delta 
            \end{align*} $$

<h2>Quantization aware Training</h2>
<p>
  General way to obtain a quantized model is to train a network with FP32 representation and quantize the resulting weights. This works well for the larger networks but fails miserably for the smaller models like Mobilenets. Common causes for such significant drop in accuracy are large differences in weights and outlier weight values. More than 100X difference in range of weights can cause weights in channels with smaller changes to have relatively high errror. Outlier weight values can make all remaning weights less precise.
</p>
<p>
    Quantization aware training models quantization during training and can provide higher results than the post training quantization. We model the quantization using simulated quantization on both weights and activations.  We use simulated weights and activations for both forward and backward passes. However we maintain floating point weights and update them with gradients. This ensure weights are updated for minor gradients and updated weights are quantized for subsequent passes through the network.
</p>
<h2>Learning Quantization Ranges</h2>
    Quantization ranges are treated differently for weight quantization and activation quantization. For weights the idea is to use a as min(w), b as max(w). For activations range depends on the inputs to the network. To estimate the ranges use the training data and then aggregate them via Exponential Moving Average(EMA). This allows network to enter a stable state where activation quantized ranges do not exclude significant range of values. In both the cases 0 is exactly represented as in integer after quantization.  
  
<h3>References</h3>
  <ol>
    <li><a href="https://www.ias.ac.in/public/Volumes/reso/021/01/0011-0030.pdf">IEEE Standard for FloatingPoint Numbers</a></li>
		<li><a href="https://nervanasystems.github.io/distiller/quantization.html">Quantization- Nervana Systems </a></li>
    <li><a href="https://petewarden.com/2016/05/03/how-to-quantize-neural-networks-with-tensorflow/">How to quantize networks with Tensorflow.</a></li>
    <li><a href="https://petewarden.com/2015/05/23/why-are-eight-bits-enough-for-deep-neural-networks/">Why are Eight Bits Enough for Deep Neural Networks?</a></li>
    <li><a href="https://arxiv.org/pdf/1806.08342.pdf">Quantizing deep convolutional networks forefficient inference</a></li>
    <li><a href="https://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf">Quantization and Training of Neural Networks for EfficientInteger-Arithmetic-Only Inference</a></li>
	</ol>
</body> 