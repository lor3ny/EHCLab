gcc -pg -Wall -D K=20 -D NUM_TRAINING_SAMPLES=40002 -D NUM_TESTING_SAMPLES=9998 -D NUM_FEATURES=100 -D NUM_CLASSES=8 -std=gnu99 -lm -o3 -c src/utils.c -o build/utils.o
gcc -pg -Wall -D K=20 -D NUM_TRAINING_SAMPLES=40002 -D NUM_TESTING_SAMPLES=9998 -D NUM_FEATURES=100 -D NUM_CLASSES=8 -std=gnu99 -lm -o3 -c src/timer.c -o build/timer.o
gcc -pg -Wall -D K=20 -D NUM_TRAINING_SAMPLES=40002 -D NUM_TESTING_SAMPLES=9998 -D NUM_FEATURES=100 -D NUM_CLASSES=8 -std=gnu99 -lm -o3 -c src/io.c -o build/io.o
gcc -pg -Wall -D K=20 -D NUM_TRAINING_SAMPLES=40002 -D NUM_TESTING_SAMPLES=9998 -D NUM_FEATURES=100 -D NUM_CLASSES=8 -std=gnu99 -lm -o3 -fopenmp -c src/knn_omp_quick.c -o build/knn.o
gcc -pg -Wall -D K=20 -D NUM_TRAINING_SAMPLES=40002 -D NUM_TESTING_SAMPLES=9998 -D NUM_FEATURES=100 -D NUM_CLASSES=8 -std=gnu99 -lm -o3 -fopenmp src/main.c build/utils.o build/timer.o build/knn.o build/io.o -o build/knn


::gcc -Wall -std=gnu99 -lm -c src/utils.c -o build/utils.o
::gcc -Wall -std=gnu99 -lm -c src/timer.c -o build/timer.o
::gcc -Wall -std=gnu99 -lm -c src/io.c -o build/io.o
::gcc -D K=3 -Wall -std=gnu99 -lm -c src/knn_omp_quick.c -o build/knn.o
::gcc -D K=3 -Wall -std=gnu99 -lm src/main.c build/utils.o build/timer.o build/knn.o build/io.o -o  build/knn