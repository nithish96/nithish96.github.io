---
title: Lottery ticket hypothesis
categories:
- Pruning
feature_text: |
  Lottery Ticket Hypothesis
feature_image: "https://picsum.photos/2560/600?image=872"
excerpt: "The main goal of neural network pruning is to reduce the size of the network complexity by removing the unwanted parts of the network. We will study one of important methods of pruning called Lottery Ticket Hypothesis"
---

<script type="text/javascript" src="http://latex.codecogs.com/latexit.js"></script>

<div>
<h3> Introduction</h3>
  <p>
  The main goal of neural network pruning is to reduce the size of the network (either by memory or speed or both) by removing the unwanted parts of the network. Main motivation for network pruning is that there are many parameters in the network in which some of them might be redundant and doesnot contribute much to the output. Network pruning removes those unwanted parts of the network either by making them zero or removing them from the network. </p>

<h3>Pruning and its types</h3>

  Depending on the parts of the network that we prune, pruning can be classified into two types as follows.
		<ol>
			<li>Structured Pruning - This involves pruning groups of elements like convolutional layer or channel. This is also known as coarse-grained pruning.</li>
			<li>Unstructured Pruning - This involves pruning individual weights based on the connection importances. This is also known as fine grained pruning.</li>
		</ol>

  Depending on the pruning schedule, pruning can be classified into two types as follows.

	<h4>Single Shot Pruning</h4>
		<p>In single shot pruning we take a trained model and prune the unwanted channels. Typically there is a drop in accuracy when we prune some of the channels. Since pruning damages the learned function by the network, fine tuning is required for the pruned model to match the accuracy of original model.
	</p>
	 <p>
		For example - Assume that we have 256 filters in particular layer i. Then if we want to remove filters that have lower magnitude (pruning criteria). Say there are 30 such filters that match pruning criteria then we remove all of these 30 filters. This reduces the number of filters in the pruned network. This way we do it for all the layers in the network there by reducing the memory footprint and the complexity of the network.
	</p>
	<h4>Iterative Pruning</h4>
	<p> It is observed that pruning followed by retraining has achieved better results than single shot pruning [Ref 4]. This is also known as Iterative pruning. Iterative pruning can be described as follows
			<ol>
				<li>Train the Network</li>
				<li>Prune the Network </li>
				<li>Retrain or finetune Network</li>
				<li>Repeat 2 and 3 iteratively</li>
			</ol>
	Retraining or finetuning step is essential for pruned network to acheive performance similar to original network. Now we gradually repeat steps 2 and 3 till the point where accuracy drops drastically. Typically we stop pruning till we achieve the required sparsity level or computational complexity. This shows that pruned networks can learn the function similar to original network but with much smaller networks. But the questions that comes out is "Why can't these networks trained from scratch ?". Today we try to look at one the papers that tries to address this.</p>
</div>

<h3>Lottery Ticket Hypothesis</h3>
<div>
	Lottery ticket hypothesis from the paper <br><br>
	<i>
	"A randomly-initialized, dense neural network contains a subnetwork that is initialized such that—when trained in isolation—it can match the test accuracy of the
original network after training for at most the same number of iterations." </i>
	<br><br>

  According to the lottery ticket hypothesis there is a subnetwork in a network that can acheive the performance of the complete network. Any standard pruning technique finds those subnetwork from original network.

  These subnetworks when initialized with initial parameters of the original network, they can acheive the performance of original network. These subnetworks are known as winning tickets since they have won initialisation lottery. If we initialize the parameters of the subnetwork randomly and train from scratch they no longer match the performance of the original network.  This tells us that smaller networks donot train effectively unless they are properly initialized.
</div>

<h4>Finding subnetworks</h4>
<div>
	<ol>
		<li>Initialise a network with initial parameters as <span lang="latex">\theta_{0} \sim D_{\theta} </span></li>
		<li>Train the network and lets say parameters are now <span lang="latex">\theta_{j}</span></li>
		<li>Prune p% of params in <span lang="latex">\theta_{j}</span> creating mask m</li>
		<li>Reset remaning params to their values in <span lang="latex">\theta_{0} </span> which creates winning ticket.</li>
	</ol>
</div>
<h4>Properties of winning tickets</h4>
<div>
	<ol>
		<li>Winning tickets learn faster than original network. Iteratively pruned winning tickets has the better generalisation than that of winning tickets that were pruned once.</li>
		<li>If we initialise winning ticket with the random initialisation they tend to learn slower than the winning ticket with original initialisation</li>
		<li>Iterative pruning is computationally expensive because we have to train network and prune the network n times. However iteratively pruned winning tickets learn faster and achieve higher test accuracy at smaller networks</li>
		<li>Since winning tickets are found using training data, it is safe to assume that structure of winning ticket has inductive bias to learning task being performed.</li>
	</ol>
</div>

<div>
	<h3>Deconstructing Lottery Ticket Hypothesis</h3>
		While training the subnetworks, it is observed that these subnetworks had accuracy significantly better than chance at initialization. Untrained network with a mask resulted in a partially working network. The resulting masks are known as supermasks. In randomly initialized networks with large final masks (weights with large magnitude are kept) it is not entirely implausible since the masks are obtained from training.
  <h4>Masking is Training</h4>
  <p>
    Masking does two things here - Zero the weights and freeze them. Authors experiment by setting the weights to initial value. If zeros doesnot matter both of them should have same accuracy. But networks seemed to perform better when the weights are frozen at zeros than random initial values.
  </p>
  <p>
    The question here is  "why is zero an ideal value?". One hypothesis here is that the mask criteria we use tend to mask to zero those weights that were headed zero anyway. To test this hypothesis authors experiment by doing following- set to zero if weights move towards 0 or set to random initial value if they mode away from zero.
  </p>
  <img src="/assets/lottery_ticket_results.png" style="width:900px;height:600px;">
  <p>
    The resulting subnetworks were performing as well as the winning tickets even when do not freeze them to 0. If we apply the same criteria to all the weights instead of pruned weights, resulting subnetworks were performing better than the winning tickets. You can see the results in the above figure. The results supports the hypothesis that subnetworks benifitted from zero was because they were moving to 0.
  </p>
  <h4>Mask criteria</h4>
	<p>This includes the set of functions that decide which weighs to prune or to keep. If we keep the weights with large magnitude in the network then criteria is named as large_final. Another criteria is magnitude increase - keep the weights that move most away from zero.  In this way authors experiment with different strategies for pruning and the results are as shown below </p>
	<img src="/assets/deconstruting_resuts.png" style="width:1000px;height:600px;">
	<!-- <p style="margin-bottom:4cm;"></p> -->
	<p>From the above figure we can infer that lottery ticket works well on different mask criteria not just only on large final weights criterion. The results shows that pruned networks can exceed original network with different mask criteria.In general, we observe that those methods that bias towards keeping weights with large final magnitude are able to uncover performant subnetworks.</p>
	<h4>Significance of initial weights</h4>
		<br>
		<p>
		Lottery ticket hypothesis says that pruned networks works better when they were rewind to their original initialisation. It turns out that pruned networks doesnot work well when they were initialised from same distribution or constant weights or reshuffle weights with in same layer. But surprisingly all of these methods works better when the initial sign of weights values in maintained. This suggests that reinitialization is not the deal breaker as long as you keep the sign. Optimizers can perform well for when weights are in correct quadrant and faces difficuly in crossing zero sign barriers. </p>
    <h4>Supermasks</h4>
    <p>
     Authors evaluate the supermasks with the single shot pruning rather than iterative pruning. They introduce new mask criteria large_final_same_sign i.e select for weights with large final magnitudes that maintained the same sign at the end of training. This resulted in networks that achi over 80% on MNIST and 24% on CIFAR without any training. Another observation is that when mask is applied with a signed constant subnetworks achieved even higher accuracy 86% on MNIST and 41% on CIFAR.
    </p>
</div>
<h3>References</h3>
<ol>
	<li><a href="https://arxiv.org/pdf/1803.03635.pdf">Lottery Ticket Hypothesis</a></li>
	<li><a href="https://nervanasystems.github.io/distiller/pruning.html">Neural Networks distiller</a></li>
	<li><a href="https://arxiv.org/pdf/1905.01067.pdf">Deconstructing Lottery Ticket Hypothesis</a></li>
  <li><a href="https://arxiv.org/pdf/1506.02626.pdf">Learning both Weights and Connections for EfficientNeural Networks</a></li>
  <li><a href="https://eng.uber.com/deconstructing-lottery-tickets/">Deconstructing Lottery Ticket Hypothesis-uber</a></li>
</ol>
