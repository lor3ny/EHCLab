gcc -Wall -D K=3 -D NUM_TRAINING_SAMPLES=12 -D NUM_TESTING_SAMPLES=5 -D NUM_FEATURES=2 -D NUM_CLASSES=2 -std=gnu99 -lm -c src/utils.c -o build/utils.o
gcc -Wall -D K=3 -D NUM_TRAINING_SAMPLES=12 -D NUM_TESTING_SAMPLES=5 -D NUM_FEATURES=2 -D NUM_CLASSES=2 -std=gnu99 -lm -c src/timer.c -o build/timer.o
gcc -Wall -D K=3 -D NUM_TRAINING_SAMPLES=12 -D NUM_TESTING_SAMPLES=5 -D NUM_FEATURES=2 -D NUM_CLASSES=2 -std=gnu99 -lm -c src/io.c -o build/io.o
gcc -Wall -D K=3 -D NUM_TRAINING_SAMPLES=12 -D NUM_TESTING_SAMPLES=5 -D NUM_FEATURES=2 -D NUM_CLASSES=2 -std=gnu99 -lm -c src/knn.c -o build/knn.o
gcc -Wall -D K=3 -D NUM_TRAINING_SAMPLES=12 -D NUM_TESTING_SAMPLES=5 -D NUM_FEATURES=2 -D NUM_CLASSES=2 -std=gnu99 -lm src/main.c build/utils.o build/timer.o build/knn.o build/io.o -o build/knn