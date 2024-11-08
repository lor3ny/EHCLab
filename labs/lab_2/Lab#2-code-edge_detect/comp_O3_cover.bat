call gcc -c -coverage IO.c
call gcc -O3 edge_detect.c IO.o -coverage -o edge_detect