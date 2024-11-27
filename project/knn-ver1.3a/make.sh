#A1a (-D DT=2 to compile considering float data types)
gcc -Wall -std=gnu99 -c utils.c timer.c io.c -lm
gcc -D DT=1 -D K=3 -Wall -std=gnu99 -c knn.c -lm
gcc -D DT=1 -D K=3 -Wall -std=gnu99 main.c knn.o utils.o timer.o io.o -o knn -lm

# A1b (-D DT=2 to compile considering float data types)
#gcc -Wall -std=gnu99 -c utils.c timer.c io.c
#gcc -D K=20 -Wall -std=gnu99 -c knn.c -lm
#gcc -D K=20 -Wall -std=gnu99 main.c utils.o timer.o knn.o io.o -o knn -lm

# A1c (-D DT=2 to compile considering float data types)
#gcc -Wall -std=gnu99 -c utils.c timer.c io.c -lm
#gcc -D K=3 -Wall -std=gnu99 -c knn.c -lm
#gcc -D K=3 -Wall -std=gnu99 main.c utils.o timer.o knn.o io.o -o knn -lm

# A2: (-D DT=2 to compile considering float data types)
#gcc -Wall -std=gnu99 -c utils.c timer.c io.c -lm
#gcc -Wall -D K=20 -D NUM_TRAINING_SAMPLES=8004 -D NUM_TESTING_SAMPLES=1996 -D NUM_FEATURES=100
#-D NUM_CLASSES=8 -std=gnu99 -c knn.c -lm
#gcc -Wall -D K=20 -D NUM_TRAINING_SAMPLES=8004 -D NUM_TESTING_SAMPLES=1996 -D NUM_FEATURES=100
#-D NUM_CLASSES=8 -std=gnu99 main.c utils.o timer.o knn.o io.o -o knn -lm

# A3:
#gcc -Wall -std=gnu99 -c utils.c timer.c io.c -lm
#gcc -Wall -D K=20 -D NUM_TRAINING_SAMPLES=40002 -D NUM_TESTING_SAMPLES=9998 -D
#NUM_FEATURES=100 -D NUM_CLASSES=8 -std=gnu99 -c knn.c -lm
#gcc -Wall -D K=20 -D NUM_TRAINING_SAMPLES=40002 -D NUM_TESTING_SAMPLES=9998 -D
#NUM_FEATURES=100 -D NUM_CLASSES=8 -std=gnu99 main.c utils.o timer.o knn.o io.o -o knn -lm
