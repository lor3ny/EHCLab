gcc -D K=20 -fno-builtin -Wno-unused-parameter -Wall -Wextra -fmax-errors=10 -std=gnu99 -lm -O3 -c src/utils.c -o build/utils.o
gcc -D K=20 -fno-builtin -Wno-unused-parameter -Wall -Wextra -fmax-errors=10 -std=gnu99 -lm -O3 -c src/timer.c -o build/timer.o
gcc -D K=20 -fno-builtin -Wno-unused-parameter -Wall -Wextra -fmax-errors=10 -std=gnu99 -lm -O3 -c src/io.c -o build/io.o
gcc -D K=20 -fno-builtin -Wno-unused-parameter -Wall -Wextra -fmax-errors=10 -std=gnu99 -lm -O3 -fopenmp -c src/knn_default.c -o build/knn_default.o
gcc -D K=20 -fno-builtin -Wno-unused-parameter -Wall -Wextra -fmax-errors=10 -std=gnu99 -lm -O3 -fopenmp src/main.c build/utils.o build/timer.o build/knn_default.o build/io.o -o build/knn_default