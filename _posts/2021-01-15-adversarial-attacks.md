---
title: Adversarial Attacks in Images
categories:
- Computer Vision
feature_text: |
  Adversarial Attacks in Images
feature_image:
   "https://picsum.photos/2560/600?image=872"
excerpt: "In this post we will look at the drawbacks of neural networks in production environments. Specifically we will look at the crafted inputs that can degrade performance of our machine learning models. We will also look at way of improving model's performance with such inputs. "

---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<body>

<div>
	<h3>Introduction</h3>
		<p>Today we see a lot of machine learning models deployed in various applications across the world. There has been an extensive amount of work done in recent times to study about the security of these deep models. Unfortunately these models are vulnerable to adversarial samples - inputs crafted by adding small amount of noise. These adversarial examples expose the fundamental dead spot in our training algorithms. We will study more about the adversarial samples and their process of generation in following sections. </p>
	<h3>Why do adversarial examples exist?</h3>
		<p>Adversarial examples are often projected as the problems of deep learning algorithms but in reality they exist across all machine learning classifiers. Resisting from adversarial examples while having a state of the art accuracy is a difficult problem and there is no fixed solution yet to this problem. There are many probable hypothesis that explains about the existence of adversarial examples. Some of them include - they exist due to the high non linear nature of neural networks or lack of regularization of deep networks. These hypothesis does not explain the existence of adversarial examples for linear models.  </p>
		<p>
			Consider an dot product with weights \( w\) and adversarial example \( x_{adv} \). This can be written as 
			\[ w^T . x_{adv} = w^T. x + w^T. \eta\] If we observe the equation perturbation causes activation to increase by a value of \( w^T . \eta \). If w had n dimensions and magnitude of element of w is m then activation will increase by \( \epsilon mn\). For the high dimensional inputs we can make many infinitesimally small changes to the input that can add up to the large change to output. This way linear model is forced to attend to the perturbed noise that aligns closely with the weights even though there are signals with the higher amplitudes. 
		</p>
		<p>Often adversarial examples generated with linear models transfer to neural networks as well i.e neural networks fails to produce correct predictions for adversarial samples. This is because neural networks are too linear to resists these attacks. This explains why simple analytical perturbations of linear models can also damage function learnt by deep networks. </p>
	<h3>Adversarial Attacks</h3>
	<p>
		Adversarial attacks can be classified into two categories - White box attacks and Black box attacks. In a white box setting given the classifier C and the sample \(x \) with label \(y\), then the goal is to synthesize fake image \(x^1 \) such that \(x^1 \) is perceptually similar to input x but can mislead the classifier C to give wrong predictions. In a Black box setting attackers can only see the outputs of the model that they are trying to attack. </p>
	<h3>Adversarial image Generation</h3>
		<h4>L-BFGS Attack</h4>
		<p> Generation of adversarial attack can be posed as an optimization problem as follows \[\text{minimize } c||x-x^1||_2^2 + L(\theta, x^1, t) \] First term imposes the similarity constraint between the input and adversarial sample. Second term tries to find \( x^1 \) that has minimal loss to label t. We keep modifying c till we can find sample that is close to current sample and can fool the classifier. To solve this optimization problem authors used LBFGS method (Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) is a non linear gradient based numerical optimization algorithm.)hence commonly known as LBFGS attack. </p>
		<h4>FGSM</h4>
		<p> This is an one step method to generate adversarial samples quickly. Let \(\theta\) be the parameters of the network, \(x\) be the input to the network with \(y\) as the target. Assume the cost function to train the network is \(J(\theta, x, y)\). Then Fast Gradient Sign Method is defined as \[ \eta = \epsilon sign (\triangledown_x J(\theta, x, y) \]</p>
		<p>Above gradient can be computed efficiently during backpropagation. This method causes a wide variety of model to misclassify their input. By using \(\epsilon = 0.25\) FGSM causes shalow softmax regression model to have an error rate of \(99.9\%\) on MNIST with a average confidence of \(79.3\).</p>
		<h4>Deep Fool</h4>
		<p>Assume f is an binary image classification network that produces \( \hat{k}(x) = sign(f(x)) \). Consider the simplest case of f i.e linear classifier of the form \(f (x) = w^T x + b \). </p>
		<p>We can interpret the distance of \(x_0 \) from hyperplane i.e F = { \( {x : w^T x + b = 0} \) }   as the robustness of the model at \( x_0\). Minimal perturbation to change the classifiers decision is nothing but the orthogonal projection of \(x_0\) onto F since perpendicular distance is the shortest distance. </p>
		<center>
			 <figure>
			  <img src="/assets/deepfool.png" style="width:700px;height:500px;" >
			  <figcaption>Image from [3].</figcaption>
			</figure>
		</center>
		<p>In practice the above algorithm often converges to a point on the hyperplane F so to push \( x_0 \) to other side of the boundary \( \hat{r} \) is multiplied with  \( 1 + \eta  \). In the paper authors chose \( \eta \) of 0.02 for the experiments. </p>
		<h4>Projected Gradient Descent</h4>
		<p>PGD finds the perturbation that maximises the loss of the model on a particular input while maintaining the size of perturbation less than epsilon. This starts with a random perutbation in \( L_p\) ball space then takes the gradient step in the direction of greater loss and projects pertubation into \( L_p\) if necessary (Move a point on to the circle if it lies outside the circle). Repeat the above step until convergence. </p>
	<h3>Adversarial Defense</h3>
		Different strategies have been proposed to resist adversarial attacks. These methods can be broadly divided into three categories 
		<ol>
			<li>Gradient Masking - Since most of the attacks are based on gradient information of the network, hiding the gradients will confound to some extent. This does prevent attacks because adversarial attacks are usually tranferable from one network to other network. </li>
			<li>Robust optimization - Retraining network can increase its robustness towards the adversarial samples. Now network will correctly classify the generated adversarial samples. </li>
			<li>Adversarial sample detection - Studies the distribution of adversarial samples and clean samples then detects adversarial samples and rejects the samples without feeding to the model. </li>
		</ol>
		<a href="/computer vision/2021/02/10/adversarial-defense/">More </a>
	<h3>Using Adversarial Attacks to improve Model performance </h3>
	<p> Adversarial attacks are often posed as a threat to machine learning models but they can also be used to improve image recognition performance. Adversarial Training (Training on only adversarial images) cannot generalize well to clean images. This can be primarily due to the distribution mismatch between the adversarial samples and clean samples. To fill the gap between the two distributions authors first run training on adversarial samples and finetune using clean samples. Results are as follows</p>
	<center>
			 <figure>
			  <img src="/assets/adv_prop_results.jpeg" style="width:600px;height:500px;" >
			  <figcaption>Image from [6].</figcaption>
			</figure>
	</center>
	<p>Authors explore a new way of using adversarial samples to improve robustness. They treat adversarial samples as new data and train networks with a mixture of adversarial and clean samples. Vanilla training using this combined data reduces the performance on clean examples. Distribution mismatch prevents the networks in learning in valuable features from both the domains. This can be solved using a auxillay batch norm design as follows </p>	
	<center>
			 <figure>
			  <img src="/assets/adv_prop.png" style="width:600px;height:500px;" >
			  <figcaption>Image from [6].</figcaption>
			</figure>
		</center>
	<p>This way combined distribution can be represented using two simpler distributions (one for clean and other for adversarial samples). This can be generalized to any number of training sample sources. For each mini batch we create adversarial samples using current state of the network and then we use BN for clean batch and auxillar BN for adversarial samples. This way all the layers except BN or auxillary BN, are optimized for both the distributions. Increase in parameters are minimal because we have same convolutional layers (which have higher number of parameters). For example - Using efficienetB7 results in an increase of 0.5% parameters. Complete pseudo code is shown below</p>
	<center>
			 <figure>
			  <img src="/assets/adv_prop_pseudocode.png" style="width:700px;height:500px;" >
			  <figcaption>Image from [6].</figcaption>
			</figure>
	</center>
	<h3>References</h3>
		<ol>
			<li><a href="https://arxiv.org/abs/1909.08072	">Adversarial Attacks and Defenses in Images, Graphs and Text: A Review</a></li>
			<li><a href="https://arxiv.org/pdf/1412.6572.pdf">Explaining and Harnessing Adversarial Examples</a></li>
			<li><a href="https://arxiv.org/pdf/1511.04599.pdf">DeepFool: a simple and accurate method to fool deep neural networks</a></li>
			<li><a href="https://medium.com/element-ai-research-lab/tricking-a-machine-into-thinking-youre-milla-jovovich-b19bf322d55c">Adversarial attacks - Element AI medium</a>
			<li><a href="https://towardsdatascience.com/know-your-enemy-7f7c5038bdf3">Know-your-enemy</a></li>
			<li><a href="https://arxiv.org/pdf/1911.09665.pdf">Adversarial Examples Improve Image Recognition</a></li>
			</li>
		</ol>
</div>
</body>
