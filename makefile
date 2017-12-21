nvcc = /usr/local/cuda-8.0/bin/nvcc
cudalib = /usr/local/cuda-8.0/lib64/
tensorflow = /usr/local/lib/python2.7/dist-packages/tensorflow/include
#include directory and library directory
CAFFE_INCLUDE = ./marginInnerProduct/
INCLUDE_DIR = /usr/include /usr/local/include $(CAFFE_INCLUDE) /usr/local/cuda-8.0/include
COMMON_FLAGS =$(foreach includedir,$(INCLUDE_DIR),-I$(includedir))

LIBS = cblas atlas boost_system boost_filesystem boost_thread cblas atlas cudart cublas curand  cuda cublas_device  
LIBS_FLAGS=$(foreach libs,$(LIBS),-l$(libs))
COMMON_FLAGS += $(LIBS_FLAGS)

LIB_DIR = /usr/local/cuda-8.0/lib64
COMMON_FLAGS += $(foreach libdir,$(LIB_DIR),-L$(libdir))

#TF_INCLUDE = /usr/local/lib/python2.7/dist-packages/tensorflow/include
$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
$(warning $(TF_INCLUDE))
#TF_LIB= /usr/local/lib/python2.7/dist-packages/tensorflow
$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
$(warning $(TF_LIB))
all: marginInnerProduct/libtf_marginInnerProduct.so
.PHONY : all

marginInnerProduct/libtf_marginInnerProduct.so: marginInnerProduct/tf_marginInnerProduct.cpp marginInnerProduct/caffe/math_functions.cpp  marginInnerProduct/caffe/common.cpp marginInnerProduct/tf_marginInnerProduct_g.o  marginInnerProduct/tf_math_function_g.o 
	g++ -std=c++11 $^ -o $@ -shared -Wl,--no-as-needed -fPIC -I $(TF_INCLUDE) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -L $(TF_LIB) -ltensorflow_framework $(COMMON_FLAGS) 


marginInnerProduct/tf_math_function_g.o: marginInnerProduct/caffe/math_functions.cu 
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 $^ -std=c++11 -c -o $@  -I $(tensorflow) -x cu -Xcompiler -fPIC -O2  $(COMMON_FLAGS)

marginInnerProduct/tf_marginInnerProduct_g.o: marginInnerProduct/tf_marginInnerProduct.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 $^ -std=c++11 -c -o $@  -I $(tensorflow) -x cu -Xcompiler -fPIC -O2  $(COMMON_FLAGS)

clean:
	rm marginInnerProduct/libtf_marginInnerProduct.so marginInnerProduct/tf_marginInnerProduct_g.o marginInnerProduct/tf_math_function_g.o

