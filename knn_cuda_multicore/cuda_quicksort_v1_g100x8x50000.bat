nvcc -D K=20 -D DT=2 -D NUM_TRAINING_SAMPLES=40002 -D NUM_TESTING_SAMPLES=9998 -D NUM_FEATURES=100 -D NUM_CLASSES=8 -O3 -c src/utils.c -o build/utils.obj
nvcc -D K=20 -D DT=2 -D NUM_TRAINING_SAMPLES=40002 -D NUM_TESTING_SAMPLES=9998 -D NUM_FEATURES=100 -D NUM_CLASSES=8 -O3 -c src/timer.c -o build/timer.obj
nvcc -D K=20 -D DT=2 -D NUM_TRAINING_SAMPLES=40002 -D NUM_TESTING_SAMPLES=9998 -D NUM_FEATURES=100 -D NUM_CLASSES=8 -O3 -c src/io.c -o build/io.obj
nvcc -D K=20 -D DT=2 -D NUM_TRAINING_SAMPLES=40002 -D NUM_TESTING_SAMPLES=9998 -D NUM_FEATURES=100 -D NUM_CLASSES=8 -O3 -c src/knn_cuda_quicksort.cu -o build/knn_cuda_quicksort_v1.obj
nvcc -D K=20 -D DT=2 -D NUM_TRAINING_SAMPLES=40002 -D NUM_TESTING_SAMPLES=9998 -D NUM_FEATURES=100 -D NUM_CLASSES=8 -O3 src/main.c build/knn_cuda_quicksort_v1.obj build/utils.obj build/timer.obj build/io.obj -o build/knn_cuda_quicksort_v1

