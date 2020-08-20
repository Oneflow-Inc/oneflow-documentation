CFLAGS = $(shell python -c "import oneflow; print(' '.join(oneflow.sysconfig.get_compile_flags()))")
LFLAGS = $(shell python -c "import oneflow; print(' '.join(oneflow.sysconfig.get_link_flags()))")
CUDAPATH = /usr/local/cuda-10.1/lib64

all: final_myop.so

myop_op.o: myop_op.cpp
	g++ -std=c++11 -c myop_op.cpp \
	-o myop_op.o                  \
	-fPIC                         \
	${CFLAGS}                     \
	${LFLAGS}                     \
	-O2

myop_cpu_kernel.o: myop_cpu_kernel.cpp
	g++ -std=c++11 -c myop_cpu_kernel.cpp \
	-o myop_cpu_kernel.o                  \
	$(CFLAGS) -fPIC

myop_gpu_kernel.o: myop_gpu_kernel.cu 
	nvcc -std=c++11 -c myop_gpu_kernel.cu \
	-o myop_gpu_kernel.o                  \
	$(CFLAGS) -x cu -Xcompiler -fPIC

final_myop.so: myop_op.o myop_cpu_kernel.o myop_gpu_kernel.o
	g++ -std=c++11 myop_op.o \
	myop_cpu_kernel.o        \
	myop_gpu_kernel.o        \
	-shared -o final_myop.so \
	$(CFLAGS)                \
	-fPIC                    \
	-L$(CUDAPATH)            \
	-lcudart                 \
	$(LFLAGS)

clean:
	rm -rf *.so *.o
