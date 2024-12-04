gcc -D K=20 -pg -o3 -fno-builtin -Wno-unused-parameter -Wall -Wextra -fmax-errors=10 -std=gnu99 -lm -c src/utils.c -o build/utils.o
gcc -D K=20 -pg -o3 -fno-builtin -Wno-unused-parameter -Wall -Wextra -fmax-errors=10 -std=gnu99 -lm -c src/timer.c -o build/timer.o
gcc -D K=20 -pg -o3 -fno-builtin -Wno-unused-parameter -Wall -Wextra -fmax-errors=10 -std=gnu99 -lm -c src/io.c -o build/io.o
gcc -D K=20 -pg -o3 -fno-builtin -Wno-unused-parameter -Wall -Wextra -fmax-errors=10 -std=gnu99 -lm -c src/knn.c -o build/knn.o
gcc -D K=20 -pg -o3 -fno-builtin -Wno-unused-parameter -Wall -Wextra -fmax-errors=10 -std=gnu99 -lm src/main.c build/utils.o build/timer.o build/knn.o build/io.o -o build/knn