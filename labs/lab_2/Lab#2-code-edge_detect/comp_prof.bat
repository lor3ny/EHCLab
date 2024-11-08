call gcc -c -pg IO.c
call gcc edge_detect.c IO.o -pg -o edge_detect
