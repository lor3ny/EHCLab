nvcc -D DT=2 -D CUDA_VERSION=2 -D K=20 -O3 -c src/utils.c -o build/utils.obj
nvcc -D DT=2 -D CUDA_VERSION=2 -D K=20 -O3 -c src/timer.c -o build/timer.obj
nvcc -D DT=2 -D CUDA_VERSION=2 -D K=20 -O3 -c src/io.c -o build/io.obj
nvcc -D DT=2 -D CUDA_VERSION=2 -D K=20 -O3 -c src/knn_cuda.cu -o build/knn_cuda_v2.obj
nvcc -D DT=2 -D CUDA_VERSION=2 -D K=20 -O3 src/main.c build/utils.obj build/timer.obj build/knn_cuda_v2.obj build/io.obj -o build/knn_cuda_v2