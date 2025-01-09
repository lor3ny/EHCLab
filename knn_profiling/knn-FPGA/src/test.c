#include "knn.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define NUM_ELEMENTS 4336
#define NUM_TESTS 1082

void generate_random_points(BestPoint dist_points[NUM_ELEMENTS]) {
    srand(time(NULL));

    for (int i = 0; i < NUM_ELEMENTS; i++) {
        dist_points[i].classification_id = rand() % NUM_CLASSES;

        dist_points[i].distance = ((float)rand() / RAND_MAX) * 4 + 2;
    }
}

int main() {

	int num_points = NUM_ELEMENTS;
	int k = K;

	BestPoint dist_points[NUM_ELEMENTS];

	//BestPoint result[K];

	generate_random_points(dist_points);
	
	select_k_nearest(dist_points, num_points, k);

	return 0;
}
