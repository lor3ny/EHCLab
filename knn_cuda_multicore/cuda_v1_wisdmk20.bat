nvcc -D K=20 -D DT=2 -O3 -c src/utils.c -o build/utils.obj
nvcc -D K=20 -D DT=2 -O3 -c src/timer.c -o build/timer.obj
nvcc -D K=20 -D DT=2 -O3 -c src/io.c -o build/io.obj
nvcc -D K=20 -D DT=2 -O3 -c src/knn_cuda_heap.cu -o build/knn_cuda_heap_v1.obj
nvcc -D K=20 -D DT=2 -O3 src/main.c build/utils.obj build/timer.obj build/knn_cuda_heap_v1.obj build/io.obj -o build/knn_cuda_heap_v1