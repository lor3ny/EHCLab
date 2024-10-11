call gcc -O3 -c -pg IO.c
call gcc -O3 edge_detect.c IO.o -pg -o edge_detect
