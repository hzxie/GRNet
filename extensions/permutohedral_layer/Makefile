# PLEASE IGNORE THIS FILE. USED FOR DEBUG ONLY.
# Created by Haozhe Xie <cshzxie@gmail.com>.

CUDA_HOME = /opt/cuda
PYTHON_PACKAGES = /usr/lib/python3.7/site-packages
PYTORCH_LIB = ${PYTHON_PACKAGES}/torch/lib

PYTHON_INCLUDE = /usr/include/python3.7m
# For PyTorch >= 1.1
PYTORCH_INCLUDE = ${PYTHON_PACKAGES}/torch/include
# For PyTorch >= 1.0
# PYTORCH_INCLUDE = ${PYTHON_PACKAGES}/torch/lib/include

# For PyTorch >= 1.2
# LD_FLAGS = -lcuda -lcudart -lcblas -lc10 -ltorch -lnvrtc
# For PyTorch >= 1.1
LD_FLAGS = -lcuda -lcudart -lcblas -lc10 -ltorch -lnvrtc -lcaffe2 -lcaffe2_gpu

CUDA_ARCH = sm_60


all: permutohedral

permutohedral: permutohedral.o math_utils.o
	nvcc -o permutohedral -I${CUDA_HOME}/include/ -I${PYTHON_INCLUDE} -I${PYTORCH_INCLUDE} -I${PYTORCH_INCLUDE}/torch/csrc/api/include -L${CUDA_HOME}/lib64 -L${PYTORCH_LIB} ${LD_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0 permutohedral_cuda.cpp  permutohedral.o math_utils.o -std=c++11

permutohedral.o:
	nvcc -c -arch=${CUDA_ARCH} permutohedral.cu -I${PYTHON_INCLUDE} -I${PYTORCH_INCLUDE} -I${PYTORCH_INCLUDE}/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11

math_utils.o:
	nvcc -c -arch=${CUDA_ARCH} math_utils.cu -I${PYTHON_INCLUDE} -I${PYTORCH_INCLUDE} -I${PYTORCH_INCLUDE}/torch/csrc/api/include -I${CUDA_HOME}/include/ -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11

clean:
	rm -rf *.o permutohedral
