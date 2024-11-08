gcc -D K=3 -pg -fno-builtin -Wno-unused-parameter -Wall -Wextra -fmax-errors=10 -std=gnu99 -lm -c utils.c timer.c io.c
gcc -D K=3 -pg -fno-builtin -Wno-unused-parameter -Wall -Wextra -fmax-errors=10 -std=gnu99 -lm -c knn.c 
gcc -D K=3 -pg -fno-builtin -Wno-unused-parameter -Wall -Wextra -fmax-errors=10 -std=gnu99 -lm main.c utils.o timer.o knn.o io.o -o knn