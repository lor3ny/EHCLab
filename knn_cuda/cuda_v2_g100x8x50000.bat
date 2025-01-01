nvcc -D CUDA_VERSION=2 -D K=20 -D DT=2 -D NUM_TRAINING_SAMPLES=40002 -D NUM_TESTING_SAMPLES=9998 -D NUM_FEATURES=100 -D NUM_CLASSES=8 -O3 -c src/utils.c -o build/utils.obj
nvcc -D CUDA_VERSION=2 -D K=20 -D DT=2 -D NUM_TRAINING_SAMPLES=40002 -D NUM_TESTING_SAMPLES=9998 -D NUM_FEATURES=100 -D NUM_CLASSES=8 -O3 -c src/timer.c -o build/timer.obj
nvcc -D CUDA_VERSION=2 -D K=20 -D DT=2 -D NUM_TRAINING_SAMPLES=40002 -D NUM_TESTING_SAMPLES=9998 -D NUM_FEATURES=100 -D NUM_CLASSES=8 -O3 -c src/io.c -o build/io.obj
nvcc -D CUDA_VERSION=2 -D K=20 -D DT=2 -D NUM_TRAINING_SAMPLES=40002 -D NUM_TESTING_SAMPLES=9998 -D NUM_FEATURES=100 -D NUM_CLASSES=8 -O3 -c src/knn_cuda.cu -o build/knn_cuda_v2.obj
nvcc -D CUDA_VERSION=2 -D K=20 -D DT=2 -D NUM_TRAINING_SAMPLES=40002 -D NUM_TESTING_SAMPLES=9998 -D NUM_FEATURES=100 -D NUM_CLASSES=8 -O3 src/main.c build/knn_cuda_v2.obj build/utils.obj build/timer.obj build/io.obj -o build/knn_cuda_v2


::gcc -Wall -std=gnu99 -lm -c src/utils.c -o build/utils.o
::gcc -Wall -std=gnu99 -lm -c src/timer.c -o build/timer.o
::gcc -Wall -std=gnu99 -lm -c src/io.c -o build/io.o
::gcc -D K=3 -Wall -std=gnu99 -lm -c src/knn_omp_quick.c -o build/knn.o
::gcc -D K=3 -Wall -std=gnu99 -lm src/main.c build/utils.o build/timer.o build/knn.o build/io.o -o  build/knn