all: permutohedral

permutohedral: permutohedral.o math_utils.o
	g++ -o permutohedral -I/opt/cuda/include/ -I/usr/lib/python3.7/site-packages/torch/include -I/usr/lib/python3.7/site-packages/torch/include/torch/csrc/api/include -L/opt/cuda/lib64 -L/usr/lib/python3.7/site-packages/torch/lib -lcuda -lcudart -lcblas -lc10 -ltorch -lnvrtc -D_GLIBCXX_USE_CXX11_ABI=0 permutohedral_cuda.cpp  permutohedral.o math_utils.o

permutohedral.o:
	nvcc -c -arch=sm_60 permutohedral.cu -I/usr/lib/python3.7/site-packages/torch/include -I/usr/lib/python3.7/site-packages/torch/include/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=0

math_utils.o:
	nvcc -c -arch=sm_60 math_utils.cu -I/usr/lib/python3.7/site-packages/torch/include -I/usr/lib/python3.7/site-packages/torch/include/torch/csrc/api/include -I/opt/cuda/include/ -D_GLIBCXX_USE_CXX11_ABI=0

clean:
	rm -rf *.o permutohedral
