---
title: Faceboxes
categories:
- Computer Vision
feature_image: "https://picsum.photos/2560/600?image=872"
feature_text: |
  Faceboxes - A Real-time Face Detector
excerpt: "Today we see a lot of neural networks that can achieve the human level performance but these neural networks are not readily deployable on edge devices due to their complexity. In this post we will look at one of such attempts towards developing face detection systems for edge devices."
---


<!-- <script type="text/javascript" src="http://latex.codecogs.com/latexit.js"></script> -->

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<body>

<br>
<div>
	<h2>Introduction</h2>
	<p>
		We see a lot of neural networks these days, that can achieve the human level performance in face detection and recognition systems. In general neural networks are trained using high performance computing machines with GPU.	But the developers who try to execute deep learning algorithms on the edge devices do not have this choice as the devices have computational constraints. So today we will look at one of such attempts towards developing face detection systems that can achieve high accuracy with performance in real time.
	</p>
</div>

<div>
	<h2>Faceboxes</h2>
	<p>
		Early methods of face detection tried to design robust features and train classifiers. They were efficient on cpu but could not work for a large variation of faces. These methods are sub optimal as the features designed are non robust and optimized each component independently. Convolutional Neural Network or CNN have been introduced as the feature extractors for  Face detection task. CNN based methods have achieved the state of art performance in detecting the faces from large variations. But they do come with the heavy computations which makes them difficult for deployment on edge devices. The former methods are known for their speed while the latter are known for their accuracy.
	</p>
	<p>
		Faceboxes is a CNN based architecture that tries to achieve high accuracy for face detection in real time. Complete architecture of the Faceboxes is shown in Figure 1. We will look into contributions of the paper first and go through each one of them in detail. Three main contributions of the paper are as follows
			<ol>
				<li>Rapidly Digested Convolutional Layers</li>
				<li>Multiple Scale Convolutional Layers</li>
				<li> Anchor Densification Strategy</li>
			</ol>
	</p>
	<img src="/assets/facebox_arch2.jpg" style="width:800px;height270px;">
	<p></p>
	<h3>Rapidly Digested Convolutional Layers</h3>
		<p>
			When we have a high resolution input image and large kernel, convolutional operation is extremely complex to run on a CPU. The main goal of these RDCL layers is to decrease the input size in the earlier layers of network.
		</p>
    <p>
      For example - Consider an Image of 224 x 224 x 3, filter of size 7 x 7 x 3 and number of filters as 64. Then number of computations of first layer are 472 Million i.e 64x(224 x 224 x 7 x 7 x 3)
    </p>
    <p>
			One of the easier ways of decreasing the input dimensionality is to use larger strides for convolution and pooling layers. Using lower kernel size in first few layers also speeds up the convolution operation. Often first convolutional layer is kept larger to alleviate the information loss that occured due to by spatial size reducing. We can have large filter size at first layer because the number of channels are only 3 for RGB image and 1 for Grayscale.
		</p>
    <p> It is observed that filters in initial layers form the opposite pairs. We take advantage of this method and use C.ReLU to increase the output dimensionality. To be precise C.ReLU has two outputs: [x,0] for positive values of input and [0,x] for negative values of input. </p>
	<h3>Multiple Scale Convolutional Layers.</h3>
		First we will discuss about Region Proposal Networks - used to generate proposals for object detection and then we talk about Multiple Scale Convolutional Layers.
		<h4> Region Proposal Networks (RPN) </h4>
			<div>
					<p>First image is fed to a convolutional network and assume that the network outputs the feature maps of the last convolutional layer. If you use VGG model then the size of feature map is 1/16th of original image.</p>
          <h5>Anchors and Anchor boxes</h5>
          <p>Each point in feature map is an anchor. We will create nine anchor boxes around the point with different scales and aspect ratios. Area of all the anchor boxes have to be the same length. By proposing various type of boxes we can generate candidate boxes that are similar to ground truth.</p>
          <p>RPN does two things - Tells whether the anchor box is a background or an object by finding IOU between the ground truth and anchor box. Find the error between the selected anchor box (IOU > 0.7 ) and ground truth. </p>
          <p>To find an error with ground truth we first calculate [x,y,width, height] and then find IOU with ground truth. RPN solves “background or object” as a binary classification and “error of ground truth” as a regression at the same time.</p>
          <p>RPN have only series of convolutional and pooling layers. Since there are no fully connected layers they can work with inputs of any size.</p>
			</div>
		<h4> Disadvantages of RPN </h4>
			<ol>
				<li>Anchors are associated with the last convolutional layers which are weak to handle variance in different scale images.</li>
				<li>Anchor associated layers have a single receptive field that cannot match the different sizes of objects in an image.</li>
			</ol>
			<h4>Multiple scale convolutional layers (MSCL)</h4>
      <p>MSCL are used to overcome the above mentioned disadvantages. Let us see MSCL achieve this.</p>
      <ol>
				<li>Multiple scale convolutions form the feature maps corresponding to different scales. If you observe in the above figure  anchors are associated to feature maps of multiple scales. This way we can have anchors generated over multiple layers with different resolutions and there by accounting for different scales of faces present in an image </li>
				<li>To avoid this problem feature maps should correspond to various sizes of receptive fields. This can be done by having convolutional filters of different sizes in the same branch like inception network. Inception is one of many ways to capture the different scale of images in cost efficient approach. Inception block used in the paper is as follows</li>
			</ol>
		<img src="/assets/facebox_arch3.jpg" style="width:400px;height:500px;">
	<h3>Anchor Densification Strategy</h3>
    <p> If you see the below image, assume there are 5 anchors and all of them have the same tiling interval. We can observe that the smaller boxes will be relatively sparser than the larger boxes. This affect the recall rate of small boxes. To solve this authors propose anchor densification strategy. The idea is to tile smaller box many times instead of doing it once.
    </p>
		<img src="/assets/anchors.jpeg" style="width:900px;height:250px;">
	<h3>Loss function</h3>
		<p>Loss function used to train is same as the multi task loss function used in training Faster RCNN[2]</p>
		$$
		\begin{align*}
		 L &= L_{cls} + L_{box} \\
  		L &= \frac{1}{N_{cls}} \sum\limits_{i} L_{cls} (p_i,p_i^*)) +\frac{\lambda}{N_{reg}} \sum\limits_{i} p_i^*L_{reg} (t_i,t_i^*)
		\end{align*}
		$$
		<p>
			<div>
				<ul>
					<li>\(p_i\) - Predicted probability of anchor i being an object <br> </li>
					<li>\(p_i^*\) - Ground truth label whether anchor i is an object <br> </li>
					<li>\(t_i\) -  Predicted cordinates <br></li>
					<li>\(t_i^*\) -  Ground truth coordinates <br></li>
					<li>\(N_{cls}\) -  Normalisation term <br></li>
					<li>\(N_{box}\) -  Number of anchor locations <br></li>
					<li>\( \lambda \) -  balancing parameter for both the terms <br></li>
				</ul>
			</div>
		</p>
	<h3>References</h3>
	<ol>
		<li><a href="https://arxiv.org/pdf/1708.05234.pdf">Faceboxes - A Real-time Face Detector</a></li>
		<li><a href="https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html#loss-function-1"> Object Recognition for dummies</a></li>
		<li><a href="https://www.quora.com/How-does-the-region-proposal-network-RPN-in-Faster-R-CNN-work">How does RPN in Faster RCNN work?</a></li>
		<li><a href="https://arxiv.org/pdf/1506.01497.pdf">Region Proposal Network (RPN)</a></li>
    <li><a href="https://mc.ai/easiest-rpn-explained-the-core-of-faster-r-cnn/">RPN explained</a></li>
    <li><a href="https://www.researchgate.net/publication/320495167_Detecting_Face_with_Densely_Connected_Face_Proposal_Network">DCFPN</a></li>
	</ol>
</div>
</body>
