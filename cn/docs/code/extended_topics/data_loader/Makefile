ONEFLOW_ROOT = /home/yaochi/oneflow
CUDAPATH = /usr/local/cuda-10.1/lib64
CFLAGS = $(shell python -c "import oneflow; print(' '.join(oneflow.sysconfig.get_compile_flags()))") -I${ONEFLOW_ROOT} -g -O0
LFLAGS = $(shell python -c "import oneflow; print(' '.join(oneflow.sysconfig.get_link_flags()))")

all: final_output.so

mini_reader_op.o: mini_reader_op.cpp mini_data_reader.h
	g++ -std=c++11 -c $< \
	-o $@                \
	-fPIC                \
	${CFLAGS}            \
	${LFLAGS}            \

mini_reader_kernel.o: mini_reader_kernel.cpp
	g++ -std=c++11 -c $<  \
	-o $@                 \
	${CFLAGS} -fPIC

mini_decoder_op.o: mini_decoder_ops.cpp
	g++ -std=c++11 -c $<  \
	-o $@                 \
	${CFLAGS} -fPIC

mini_decoder_kernel.o: mini_decoder_kernels.cpp
	g++ -std=c++11 -c $<  \
	-o $@                 \
	${CFLAGS} -fPIC

final_output.so: mini_reader_op.o mini_reader_kernel.o mini_decoder_op.o mini_decoder_kernel.o
	g++ -std=c++11 $^        \
	-shared -o final_output.so \
	${CFLAGS}                \
	-fPIC                    \
	-L${CUDAPATH}            \
	-lcudart                 \
	${LFLAGS}

clean:
	rm -rf *.so *.o
