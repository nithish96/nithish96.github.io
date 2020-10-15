---
title: On Calibration of Modern Neural Networks
categories:
- Neural Networks
feature_text: |
  On Calibration of Modern Neural Networks
feature_image:
   "https://picsum.photos/2560/600?image=872"
excerpt: "In this post we will look at some of the major differences between how deep neural networks classify objects and humans classify objects. We will look at some of the ways to calibrate neural networks so they can could be interpreted by humans."

---
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<body>

<div>
	<h3>Introduction</h3>
		<p>
			Deep neural networks have achieved human level performance on a variety of tasks. There has been area of deep learning (Adversarial Attacks) that has shown that - small perturbations in input can cause model to output an another label with high confidence. For humans it is natural to recognize the images that are slightly off from the original like blurred images, shifted and so on. In case of unrecognizable images like random noise humans can easily reject those samples while deep learning models tend to predict those images with high confidence. 
		</p>
		<center>
			<figure>
				<img src="/assets/fooled_images.jpg" style="width:500px;height:500px">	
				<figcaption>1. Image taken from "Deep Neural Networks are Easily Fooled"</figcaption>
			</figure>
		</center>
		<br>
		<p>
			If we look at the above image, they are completely unrecognisable to humans but the deep learning models have predicted with high confidence (around 99.6%). This can be a potential problem in applications related to health care, autonomous driving etc. This happens because when an image is fed to the model it is forced to predict from one of the trained classes (i.e. no unknown class).   
		</p>
		<p>
			We could be better off by saying model is uncertain about its decision. so that we can raise a warning in safety critical applications. But the question is how can we capture this uncertainity? First let us look at the types of uncertainity	</p>
			<ol>
				<li><b>Aleatoric Uncertainity</b> - Also known as statistical uncertainity. It is a representative of unknowns and hence differs each time we run the experiment with same setting. This cannot be reduced even with an infinite amount of data. </li>
				<li><b>Epistemic Uncertainity</b> - Also known as systematic uncertainity. It is due to things that one could know in principle know but not in practice. This happens in cases like "measurement is inaccurate because something is not calibrated". </li>
			</ol> 
	<h3>Calibration</h3>
			<p>Measures how well predicted confidence aligns with the observed accuracy. Let us say we have predicted 100 images with probability of 0.8 then we can expect 80% of the decisions are correct. If the neural networks are perfectly calibrated then caliration value would be around 0.8 but unfortunately it is not. Deep neural networks often predict high confidences for the images than traditional models. For example - Predictions made by Resnet architecture on CIFAR 100 are shown in below figure (top). We can observe that most of the predictions or confidence scores lie between 0.9 and 1.  </p>
		<center>
			<figure>
				<img src="/assets/calib.png" style="width:600px;height:600px">	
				<figcaption>2. Image taken from "On Calibration of Modern Neural Networks"</figcaption>
			</figure>
		</center>
		<p>
			 Reliability diagrams are used for the visual representation of model calibration. In reliability diagram the ideal confidence would be a diagonal line on a graph drawn with accuracy vs confidence as depicted in above figure (below). In reliability diagram if confidence values lie below the diagonal line then models are overfitted and if they lie above diagonal line they are underfitted. Deep networks like resnet produce higher confidence scores for incorrect inputs which affects their accuracy value in that interval and hence they lie below the diagonal line.   
		</p>
		<p><b>Expected Calibration Error (ECE)</b> - Difference between the confidence and accuracy. ECE is calculated by paritioning the confidence values into M bins and taking the weighted average of bins. \[ ECE = \sum\limits_{m=1}^{M} \frac{|B_m|}{n} |acc(B_m) - conf(B_m)|\]</p>
		<p><b> Negative Log Likelihood </b>- It is same as cross entropy loss but with a different interpretation of the same formula. Since entropy is a measure of uncerainity it can be used to measure the quality of probabilistic models.  Given a probabilistic model \( \hat{\pi} (Y | X) \) and n samples,  NLL is defined as \[ L = -\sum\limits_{i=1}^{n} log(\hat{\pi} (y_i | x_i))\]</p>
		<h4>Causes of Miscalibration</h4>
			<br>
			<ol>
			<li style="padding-bottom:1em"> <b>Model Capacity</b> - Today it is common to see the deep networks with hundreds of layers and thousands of filters. Although increasing depth and width have shown to produce accurate results this negatively affects the model calibration. Though smaller networks have miscalibration ECE increases with the model capacity. </li>
			<li style="padding-bottom:1em"> <b>Batch Normalization</b> - Batchnorm increases the training stability of deep networks by normalizing activations within the network. Empirically it is found that networks trained with batchnorm have higher miscalibration than their counterparts. </li>
			<li style="padding-bottom:1em"><b>Weight decay </b>- This is kind of regularization technique that has been used to train deep networks. Due to the regularization effects of batch normalization it is uncommon to see the networks with high weight decay. Training with less weight decay has negative impact on calibration. Model calibration continues to improve when regularization is added after the point of achieving optimal accuracy. </li>
			</ol>
	<h3>Calibration Methods</h3>
		<p>All the methods described here are post processing steps that convert the predicted probabilities to calibrated probabilities. Calibration methods requires the data to perform calibration (can be the same set of data that is used for hyperparameter tuning). There are two kinds of calibration methods - parametric methods and non parametric methods. </p>
		<h4>Calibrating Binary Models</h4>
			<p>
				Let us assume a binary classfication system - \( p_i\) is the predicted probability for a given input \( x_i\)  and \( y_i\) is the ground truth label. Our goal is to compute calibrated probability \(q_i\) from predicted probability \(p_i\)(i.e Generally obtained from sigmoid).
			</p>
			<ol>
				<li style="padding-bottom:1em"><b>Histogram binning</b> - is an example of non parametric calibration methods. In this we would divide the predicted confidence scores into equally spaced bins. During the test time,  bin is identified for test input based on confidence score and the average number of positive samples in that bin is returned as the calibrated probabilty</li>
				<li style="padding-bottom:1em"><b>Isotonic Regression</b> - is also an example of non parametric calibration methods. In this method we learn a piecewise constant function f that transforms the predicted probability to calibrated probability \(f(p_i) =  q_i\). f is optimized by minimizing the square loss between the \(\sum\limits_{i=1}^{n} (f(p_i) - q_i)^2 \). Isotonic regression is a generalization of histogram binning. Formally this can be defined as</li>
				<center>
					<figure>
						<img src="/assets/isotonic.jpg" style="width:500px;height:220px">
					</figure>
				</center>
				<p>where M is the total number of bins, \( a_1, a_2, ... a_m \) are the boundaries of the intervals and \( \theta_1, \theta_2 .. \theta_m\) are the calibrated probabilities. During the optimization process the boundaries and predcition values are jointly optimzied.  Assume that the interval of first bin corresponds to [0, 0.2] and the accuracy of the first bin is 0.15. Then any test input with predicted score between 0 and 0.2 would be given the probability of 0.15</p>
				<li><b>Bayesian binning into Quantiles (BBQ) </b> - extends the histogram binning by considering multiple binning models and their combinations. Different binning models differ in the number of bins they have. This method considers the space S of all binning schemes. BBQ performs the bayesian average of probabilities produced by each scheme. </li>
				\[ \begin{align*}
					P(\hat{q}_{te} | \hat{p}_{te}, D) &= \sum\limits_{s \in S}^{} P(\hat{q}_{te}, S=s | \hat{p}_{te}, D) \\
													  &= \sum\limits_{s \in S}^{} P(\hat{q}_{te}| \hat{p}_{te}, S=s,  D) P(S=s | D) 
					\end{align*}\]
					<p>where \( P(\hat{q}_{te}| \hat{p}_{te}, S=s,  D)\) is a calibrated probability with respect to binning scheme s. We can compute calibrated probability \( P(\hat{q}_{te} | \hat{p}_{te}, D)\) for any test input.</p>
				<li><b>Platt Scaling</b> is a parametric approach to calibration. In this method logistic regression is trained on the logits of model to compute calibrated probabilities \( q_i = \sigma(a. z_i + b) \). a and b are optimized by using NLL loss over validation set. This doesnt affect original models accuracy since all the parameters of the network remains same. </li>
			</ol>
		<h4>Extension to Multiclass Models</h4>
			<ol>
				<li style="padding-bottom:1em"><b>Extension of binning methods</b> - One common way of extending binning method is to use one vs all strategy for each particular class. This extension can be applied to histogram binning, isotonic binning and also BBQ. </li>
				<li style="padding-bottom:1em"><b>Matrix and Vector Scaling</b> - applies a linear transformation to logits i.e \( W. z_i + b\). Parameters W and b are optimzied with NLL on validation set. Number of parameters grows quadratically with number of classes. Vector scaling is variant of matrix scaling where W is restricted to be a diagonal matrix. </li>
				<li><b>Temperature Scaling</b> - is an simple extension of platt scaling. A single parameter called T is used for all classes. Logits \(z_i\) are divided by T before feeding to the softmax layer. It softens the softmax with T > 1. As T-> \(\infty \) \(\hat{q}_i\) becomes \( \frac{1}{K}\).  As T -> 0 porbability collapses to a point mass \( \hat{q}_i = 1\). T is optimized with respect to NLL on validation set. Since temperature scaling doesnt change the maximum of softmax function accuracy will not get affected. </li>
			</ol>
	<h3>Feasibility </h3>
			Temperature scaling is the fastest method to achieve calibration and can easily be incorporated into training pipeline. We set T=1 during training and find optimal value on validation set. Histogram binning and isotonic binning take order of magniture longer than temperature scaling. BBQ is difficult to implment and takes three orders of magnitude more time. 
	<h3>References</h3>
	<ol>
		<li><a href="https://arxiv.org/pdf/1412.1897.pdf">Deep Neural Networks are Easily Fooled</a></li>
		<li><a href="https://arxiv.org/pdf/1706.04599.pdf">On Calibration of Modern Neural Networks</a></li>
		<li><a href="https://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf">Obtaining Well Calibrated Probabilities Using Bayesian Binning</a></li>
		<li><a href="http://ethen8181.github.io/machine-learning/model_selection/prob_calibration/prob_calibration.html">Probability Calibration</a></li>
		<li><a href="https://en.wikipedia.org/wiki/Uncertainty_quantification">Uncertainity Quantification</a></li>
	</ol>

</div>
</body>