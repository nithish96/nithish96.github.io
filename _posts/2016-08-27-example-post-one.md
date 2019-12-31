---
title: Faceboxes for face detection
categories:
- Face detection
feature_image: "https://picsum.photos/2560/600?image=872"
---


<script type="text/javascript" src="http://latex.codecogs.com/latexit.js"></script>
<br>
<div>
	<h2>Introduction</h2>
	<p>
		Today we see a lot of neural networks that can achieve the human level performance in face detection and recognition systems. In general these kind of neural nets are trained using high performance computing machines that use GPUS.	But the developers who try to execute deep learning algorithms on the edge devices do not have this choice. So today we will look at one of such attempts towards developing face detection systems that can achieve high accuracy and performance at the same time.
	</p>
</div>

<div>
	<h2>Faceboxes</h2>
	<p>
		Early methods of face detection were done Using hand crafted features. These methods are non robust are they are independently trained for particular components. Face detection have also been motivated from the object detection based methods like RCNN. CNN based methods have achieved the state of art performance in detecting the faces from large variations. But they do come with the heavy computations which makes them difficult for deployment on the edge devices. The former methods are known for their speed and while the latter are known for their accuracy. 
	</p>
	<p>
		Complete architecture of the model is shown in Figure 1. In this section we will look into contributions of the paper first and go through each one of them in detail. Three main contributions of the paper are as follows
			<ol>
				<li>Rapidly Digested Convolutional Layers</li>
				<li>Multiple Scale Convolutional Layers</li>
				<li> Anchor Densification Strategy</li>
			</ol>
	</p>
	<img src="/facebox_arch.jpeg" style="width:1000px;height:200px;">
	<p></p>
	<h3>Rapidly Digested Convolutional Layers</h3>
		<p>
			When we have high resolution input image and large kernel convolutional operation is extremely costly to run on a CPU. The main goal of these RDCL layers is to decrease the input size as follows
		</p>
		<p>
			One of the easier ways of decreasing the input dimensionality is to use larger strides for convolution and pooling layers. Using lower kernel size in first few layers also speeds up the convolution operation. Often first convolutional layer is kept larger to alleviate the information loss that occured due to by spatial size reducing. 
		</p>
		<p> It is observed that filters in initial layers form the opposite pairs. Original output channels were decreased and then we use the C.ReLU to increase the output dimensionality. To be precise C.ReLU has two outputs: [x,0] for positive values of input and [0,x] for negative values of input. </p>
	<h3>Multiple Scale Convolutional Layers</h3>
		<p>First we will see how things are done without mult scale convolutions and then analyse the disadvantages of it. Finally how can multi scale convolutions overcome them. </p>
		<h4> Region Proposal Networks (RPN) </h4>
			<div>
				<ul>
					<li>First image is fed to a convolutional network and assume that the network outputs the feature maps of the last convolutional layer</li>
					<li>We run a sliding window of n*n across feature maps. At each step set of anchors (typically 9) are generated around the center of sliding window. Then Intersection Over Union (IOU) is computed for the each of the generated boxes with the ground truths.</li>
					<li>Boxes generated are fed to the network that has two parts - classifier and regressor. Classifier outputs whether the box contains an object (1 if box contains an object or 0 if it does not)  or not whereas the regresssor outputs the predicted bounding box coordiinates</li>
				</ul>
			</div>
		<h4> Disadvantages of RPN </h4>
		<p>
			<ol>
				<li>Anchors are associated with the last convolutional layers which are weak to handle variance in different scale images.</li>
				<li>Anchor associated layers have a single receptive field that cannot mactch the different sizes of image in image.</li>
			</ol>
		</p>
		<p>
			Multiple scale convolutional layers are used to overcome the above mentioned disadvantages. Let us see how we can achieve this.
			<ol>
				<li>Multiple scale convolutions form the feature maps corresponding to different scales. Next step is to associate the anchors to feature maps obtained from multi scale convolutions. This way we can take care of different scales of faces present in an image </li>
				<li>To avoid the problem of single receptive field in the network we can use the inception like networks. Advantage of these networks is that they have convolutions performed with different scales. This way we can capture the different scale of images in cost efficient approach. </li>
			</ol>
		</p>
		<img src="/MSCL.png" style="width:600px;height:500px;">
	<h3>Anchor Densification Strategy</h3>
		<p> Comparing with the large anchors small anchors are relatively sparse. This effects the recall rate of small faces. Inorder to solve this problem the authors propose anchor densification strategy. Idea of this method is to densify smaller anchors so an anchor is tiled n<sup>2</sup> times instead of doing it once. </p>
	<h3>Loss function</h3>
		<p>Loss function used to train is same as the multi task loss function used in training Faster RCNN[2]</p>
		<div lang="latex">
			\centerline{ L = L_{cls} + L_{box} }
		</div>
		<div lang= "latex" >
			L = \frac{1}{N_{cls}} \sum\limits_{i} L_{cls} (p_i,p_i^*) + \lambda \frac{1}{N_{reg}} \sum\limits_{i} p_i^*L_{reg} (t_i,t_i^*)
		</div>	
		<p>
			<div>
				<ul>
					<li><span lang="latex">p_i - </span>Predicted probability of anchor i being an object <br> </li>
					<li><span lang="latex">p_i^* - </span>Ground truth label whether anchor i is an object <br> </li>
					<li><span lang="latex">t_i - </span> Predicted cordinates <br></li>
					<li><span lang="latex">t_i^* - </span> Ground truth coordinates <br></li>
					<li><span lang="latex">N_{cls} - </span> Normalisation term <br></li>
					<li><span lang="latex">N_{box} - </span> Number of anchor locations <br></li>
					<li><span lang="latex">p_i - </span> balancing parameter for both the terms <br></li>
				</ul>
			</div>
		</p>
	<h3>References</h3>
	<ol>
		<li><a href="https://arxiv.org/pdf/1708.05234.pdf">Faceboxes - A Real-time Face Detector</a></li>
		<li><a href="https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html#loss-function-1"> Object Recognition for dummies</a></li>
		<li><a href="https://www.quora.com/How-does-the-region-proposal-network-RPN-in-Faster-R-CNN-work">How does RPN in Faster RCNN work?</a></li>
		<li><a href="https://arxiv.org/pdf/1506.01497.pdf">Region Proposal Network (RPN)</a></li>
	</ol>
</div>
