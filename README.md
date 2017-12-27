# A re-implementation of sphereface. 

### Introduction
The operation of user-defined op(marginInnerProduct) is completely the same as it in the sphereface-caffe.

### Requirements
Tensorflow
boost cuda
matlab
### Installation
   1. In the root direcotory
      Change the tensorflow path in makefile
      Then
	```Shell
	make
	```

#### Train
   1. Change the configuration in config.py.
   2. 
	```Shell
	python train
	```

#### Test(LFW)
   1. Put the aligned face image to "lfw_evaluation" folder
   2. Change the path in get_lfw_features.py
	```Shell
	python get_lfw_features.py
	cd ../lfw_evaluation
	run evaluation
	```

#### Result
   1. I've get 98.10 accuracy on LFW. I believe a better result will get done if I use data augmentation(mirror, smooth, jpeg compression and so on)
