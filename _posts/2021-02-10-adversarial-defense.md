---
title: Adversarial Defense in Images
categories:
- Computer Vision
feature_text: |
  Adversarial Defense in Images
feature_image:
   "https://picsum.photos/2560/600?image=872"
excerpt: "In this post we will look at the some of the approaches to resist against adversarial attacks. We will also look at some of the reasons why we donot yet have a strong defense mechanism that is robust towards all kinds of attacks. "

---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>	
<body>

<div>
	<h3>Introduction</h3>
	<p>We have covered adversarial attacks in our previous <a href="/computer vision/2021/01/15/adversarial-attacks/">post</a>. Today we will look at some of the countermeasures to resist adversarial samples. We donot yet have any strong defense method that works against all adversarial attaks. As seen in previous post adversarial defenses can be broadly classified into three categories - Gradient Masking, Robust optimization and adversarial sample detection. Let us look at some of the methods in these three categories in following sections</p>
	<h3>Gradient Masking</h3>
		<p>Since most of the attacks are based on gradient information of the classifier defender tries to hide the gradient information. Some of the examples of gradient masking are described in the following sections </p>
		<h4>Defensive Distillation</h4>
			<p>Distillation is the strategy that is initially designed to reduce the size of deep networks by training two networks. First network is known as teacher network and second network is known as student network or distilled network. Typically distilled network is compressed network than a teacher network. The labels to distilled network are called soft labels (predicitons of a teacher network). Distillation works because it smoothens the decision surface in adversarial directions exploited by adversary. Probability distributions encoded using a teacher network have more information than the hard labels. For example - In MNIST we can observe that 3 is more similar to  8 than 1.</p> 
		<p> Distillation can be reformulated to train a network that can resist adversarial samples. Defensive distillation has a different goal ( robustness towards adversarial samples) to make decision surfaces smooth. So it works even if the teacher network and student network are of same size. 	Defensive Distillation process in defined as follows</p>
		<ol>
			<li>Train a network F with training set (X, Y) by using temperature T and evaluate the model at temperature T</li>
			<li>Train a another network \(F^1 \) using training set (X, F(X)) at temperature T and do predictions on model at temperature T=1. </li>
		</ol>
		<p> In defensive distillation student network is same as teacher network. When we train a distilled network at temperature T and test it at T=1 we get outputs scaled by a factor of T. This will output softmax scores as target class being close to 1 and other class scores close to 0. We can think of softmax output similar to that of an indicator function in this scenario. This way we cannot have a gradient of a score function since it is not differentiable at that point. 	</p>
		<h4>Stochastic Gradients</h4>
		<p>We train a set of classifiers \( s = {F_t : t = 1, 2 ... k} \). During an evaluation phase we randomly pick a classifier and predict the outcome. Since attacker is not aware of the exact model that is being used for evaluation attack,  success rate can be minimized. We can drop some neurons in each layer and see if we get same results. Enabling dropout at test time has similar interpretation of training ensemble models.  </p>
		<h4>Vanishing Gradients</h4>
		<p>PixelDefend and DefenseGAN use generative networks that can convert adversarial sample to real sample. PixelDefend uses pixelCNN generative model and Defense GAN uses GAN architecture. Both of these models add a generative network before classifier resulting in extremely deep network. These defenses succeed because as the depth increases the gradient \( \frac{\partial L(x)}{\partial x}\) decreases. This prevents the attacker from learning the exact location of adversarial samples. </p>
		<h4>Disadvantages</h4>
		<p>Main problem with gradient masking strategies is that they donot eliminate the existence of adversarial samples. Usually adversarial attacks generated using one model are transferable to other models. </p>
	<h3>Robust optimization</h3>
		<p>This method tries to improve the classifier by changing the way of learning parameters. Here the major focus is to learn the parameters that minimize the adversarial loss or to learn the parameters that maximize the average minimal perturbation distance. Typically defender assumes a prior knowledge of adversarial space D and build classifiers which are safe against these attacks. </p>
		<h4>Adversarial Training</h4>
		<p>This strategy uses adversarial samples crafted with FGSM along with true labels as a additional augmentation. Trained model will now predict the future adversarial samples correctly. Using batch norm in networks helps in efficiency of adversarial training because adversarial samples come from a different distribution. Trained model will have good robustness to FGSM but it is still vullnerable to other attacks. </p>
		<p>Extension of this work is to use PGD attack instead of single step attack. PGD attacks usually find the most adversarial sample in \( l_{\infty}\) ball around x. This trains only on adversarial samples and hence helps in learning the parameters that minimize the adversarial loss. Since PGD is an iterative attack the time complexity of this training is k times that if natural training (generate adversarial sample using iterative attack for each training sample). As a result it is difficult to scale to larger datasets like Imagenet. </p>
		<center>
			 <figure>
			  <img src="/assets/adv_training_pgd.png" style="width:700px;height:400px;" >
			  <figcaption>Fig-1: Adversarial training using PGD. Image from [1].</figcaption>
			</figure>
		</center>
		<p>Another extension is to first train a set of classifiers F1, F2, F3. Then for each training sample generate an adversarial samples using F1, F2, F3. Because of transferability property they are likely to mislead our current network F as well. Training on these samples will minimize the adversarial loss. This is more efficient process than the previous two because it decouples the two process - generating adversarial samples and model training. </p>
	<h3>Adversarial Example Detection</h3>
		<p>Instead of predicting the models input directly this method tries to classify whether the input is benign or an adversarial. If its an adversarial input then model will refuse to predict its label. </p>
		<h4>Auxillary Model</h4>
		<p>One way is to train a model with K+1 classes where k+1 class consists of all adversarial samples. We can train a binary classifier to detect adversarial/benign image and then train a classifier on reocognized clear images. </p>
		<h4>Using statistics</h4>
		<p>Some early works in this area study the distribution of adversarial samples and distinguish from the clear samples. It is observed natural images have higher weight on early principle components and adversarial samples have higher weights on larger principle components. MMD (Maximum Mean Discrepency) is used to test whether two samples are drawn from the same distribution. We can use MMD to test if group of points belong to benign or adversarial set. </p>
		<h4>Checking prediction consistency</h4>
		<p>Some of the works focus on checking the consistency of the sample's prediction outcome. They usually manipulate the model parameters or input and then check if outcome have significant changes. These are based on the assumption that natural images give same results with these manipulations. </p>
		<h4>Disadvantages</h4>
		<p>Carlini Wagner attack (CW attack) has surpassed 10 of the detection methods that falls into the above three categories. They hypothesize that properties that are intrinsic to adversarial samples are hard to find. </p>
	<h3>Why is to hard to defend?</h3>
		<ol>
		<li>Adversarial examples are hard to defend there is no theoretical or well defined way of crafting adversarial samples. We dont have a good theoretical tools to find solutions to the complicated optimization problems so it is hard to find a defense that rules out a set of adversarial examples.</li>
		<li> From a different perspective, adversarial examples are hard to defend because they require models to give same result for every possible input. Most of the time models work only with the small amount of data from many possible input they might encounter. Because of the number of possible inputs it is hard to find a defense that is truly adaptive. </li>
		</ol>
	<h3>References</h3>
	<ol>
		<li><a href="https://arxiv.org/abs/1909.08072">Adversarial Attacks and Defenses in Images, Graphs and Text: A Review</a></li>
		<li><a href="http://www.cleverhans.io/security/privacy/ml/2017/02/15/why-attacking-machine-learning-is-easier-than-defending-it.html">Is attacking machine learning easier than defending it?</a></li>
	</ol>

</div>
</body>
