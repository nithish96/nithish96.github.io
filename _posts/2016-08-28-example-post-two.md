---
title: Lottery ticket hypothesis
categories:
- Pruning
feature_text: |
  Neural Network Pruning
---

<script type="text/javascript" src="http://latex.codecogs.com/latexit.js"></script>

<h3> What is Neural Network pruning</h3>	
<div>
	The main goal of neural network pruning is to reduce the size of the network complexity (either by both size and speed) by removing the unwanted parts of the network. The idea of network pruning is that there are many parameters in the network in which some of them might be redundant and doesnot contribute much to the output. Typically we stop pruning till we achieve the required sparsity level or computational complexity. There are mainly two kinds of pruning
	<br>
	<br>
		<ol>
			<li>Structured Pruning - This involves pruning groups of elements like convolutional layer or channel. This is also known as coarse-grained pruning.</li>
			<li>Unstructured Pruning - This involves pruning individual weights based on the connection importances. This is also known as fine grained pruning.</li>
		</ol>	
	<h4>Single Shot Pruning</h4>
		In single shot pruning we take a trained model and prune the unwanted channels. Typically there is a drop in accuracy when we prune some of the channels. Fine tuning is required for the pruned model to match the accuracy of original model.
	<br><br>
	<p>
		For example Assume that we have 256 filters in particular layer i. Then if we remove filters that have lower magnitude (below particular threshold). Say if there are 30 such filters that match pruning criteria we remove all of them. This reduces the number of filters in the pruned network. This way we do it for all the layers in the network there by reducing the memory footprint and the complexity of the network.
	</p>
	<h4>Iterative Pruning</h4>
		Iterative pruning can be described as follows
		<ul>
			<li>Train the Network</li>
			<li>Prune Network</li>
			<li>Retrain or finetune Network</li>
			<li>Repeat 2 and 3 iteratively</li>
		</ul>
	<p>
	Pruning the network damages the function that we have learnt. So third step is essential for pruned network to acheive performance similar to original network. Now we gradually repeat steps 2 and 3 till the point where accuracy drops drastically. This shows that pruned networks can learn the function similar to original network but with much smaller networks. But the questions that comes out is "Why cant these networks trained from scratch ?". Today we try to look at one the papers that tries to address this.
	</p>
</div>

<h3>Lottery Ticket Hypothesis</h3>
<div>
	Lottery ticket hypothesis from the paper
	<b>
	"A randomly-initialized, dense neural network contains a subnet-
work that is initialized such that—when trained in isolation—it can match the test accuracy of the
original network after training for at most the same number of iterations." </b>
	<br><br>
	<p style="margin-bottom:0.3cm;">Mathematically it can be stated as </p>
	<p>Let us say we have a neural network <span lang="latex"> f(x;\theta_{0}) </span> which achieves minimal validation loss L1 at iteration j with test accuracy A1. If we train a network <span lang="latex"> f(x;m  \odot \theta_{0}) </span> with the mask m <span lang="latex"> m \in (0,1)^{|\theta|} </span> such that initialisation is <span lang="latex"> m  \odot \theta_{0} </span>, then on the same training data this network achieves minimum validation loss at iteration k with test accuracy A2. Then lottery ticket hypothesis says k <= j and A2>=A1 and <span lang="latex"> ||m||_0 << |\theta| </span></p>
</div>

<div>
	<h4>How to find winning tickets</h4>
	<ol>
		<li>Initialise a network with initial parameters as <span lang="latex">\theta_{0} \sim D_{\theta} </span></li>
		<li>Train the network and lets say parameters are now <span lang="latex">\theta_{j}</span></li>
		<li>Prune p% of params in <span lang="latex">\theta_{j}</span> creating mask m</li>
		<li>Reset remaning params to their values in <span lang="latex">\theta_{0} </span> which creates winning ticket.</li>
	</ol>
	<h4>Properties of winning tickets</h4>
	<ol>
		<li>Winning tickets learn faster than original network. Iteratively pruned winning tickets has the better generalisation than that of winning tickets that were pruned once.</li>
		<li>If we initialise winning ticket with the random initialisation they tend to learn slower than the winning ticket with original initialisation</li>
		<li>Iterative pruning is computationally expensive because we have to train network and prunt the network n times. However iteratively pruned winning tickets learn faster and achieve higher test accuracy at smaller networks</li>
		<li>Since winning tickets are found using training data, it is safe to assume that structure of winning ticket has inductive bias to learning task being performed.</li>
	</ol>
</div>

<div>
	<h3>Deconstructing Lottery Ticket Hypothesis</h3>
		Performance of pruned networks often exceeds or equals the accuracy of the original network for the reasons that were initially well defined. In [3] they tried to study the three main critical main components of the lottery ticket hypothesis and show why pruned model tends to give such high performance.
	<br><br>
	<h4>Mask criteria</h4>
	<p>This includes the set of functions that decide which weighs to prune or to keep. If we keep the large weights in the network then criteria is named as large_final. In this way authors experiment with different strategies for pruning and the results are as shown below </p>
	<img src="/deconstruting_resuts.png" style="width:1000px;height:600px;">
	<!-- <p style="margin-bottom:4cm;"></p> -->
	<p>From the above figure we can infer that lottery ticket works well on different mask criteria not just only on large final weights criterion. This also says the pruned networks can exceed original network with different mask criteria.</p>
	<h4>Significance of initial weights</h4>
		Lottery ticket hypothesis says that pruned networks works better when they were rewind to their original initialisation. It turns out that pruned networks doesnot work well when they were initialised from same distribution or constant weights or reshuffle weights with in same layer. But surprisingly all of these methods works better when the initial sign of weights values in maintained. Optimizers can perform well for when weights are in correct quadrant and faces difficuly in crossing zero sign barriers.
	<br><br>
	<h4>Masking is training</h4>
		In general pruned weights are set to 0 as these are not important for the network. If these weights are no important then setting them to a constant value or initial value should work. But it doesnot zero values matter and helps to achieve better performance of the networks. The intuition of this is as follows -> Mask criteria would mask weights that tend to move zero anyway
		<br>
		<br> 
</div>
<h3>References</h3>
<ol>
	<li><a href="https://arxiv.org/pdf/1803.03635.pdf">Lottery Ticket Hypothesis</a></li>
	<li><a href="https://nervanasystems.github.io/distiller/pruning.html">Neural Networks distiller</a></li>
	<li><a href="https://arxiv.org/pdf/1905.01067.pdf">Deconstructing Lottery Ticket Hypothesis</a></li>
</ol>