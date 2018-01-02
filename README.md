# A re-implementation of sphereface. 

### Introduction
The operation of user-defined op(marginInnerProduct) is completely the same as it in the sphereface-caffe. 

So the operation of computing Backward gradient also uses the tricks [sphereface](https://github.com/wy1iu/sphereface) used.

Thanks to all the contributors of [sphereface](https://github.com/wy1iu/sphereface) and [caffe](https://github.com/BVLC/caffe)


### Data augmentation
Now I provide the c++ code of some useful data augmentation(shift, zoom, rotation, modHSV, modRGB and so on) in data_augmentation.hpp.


### Requirements
	Linux (I've test on ubuntu16.04)
	Tensorflow1.4+(?)
	boost cuda(if you have installed caffe, there is no problem  for those libraries)
	matlab
### Installation
   1. First:

	git clone https://github.com/hujun100/tensorflow-sphereface.git 	
	cd tensorflow-sphereface

   2. Change the path variable TENSORFLOW and NVCC in "makefile" according to your configuration
      Then:

	make


### Train
   1. Change the configuration in config.py.
   2. 

	cd train_test
	python train.py

### Test(LFW)
   1. Put the aligned face images to "lfw_evaluation" folder
   2. Change the test configuration in config.py
   3. Change the test_data_dir in evaluation.m

	python get_lfw_features.py
	cd ../lfw_evaluation
	run evaluation.m

### Result(LFW)
   1. I've get 98.10 accuracy on LFW trained on CASIA-webface. I believe a better result will be obtained if I use data augmentation(mirror, smooth, jpeg compression and so on)

### Reference 
[sphereface](https://github.com/wy1iu/sphereface)

[caffe](https://github.com/BVLC/caffe)
