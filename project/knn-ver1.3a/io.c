/**
	Functions to read values from .dat files and to generate random data
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>

#include "io.h"

void read_data(int argc, char **argv,
				int *num_points, int *num_classes, int *num_new_points, int *k,
				Point **known_points, Point **new_points,
				char **key, int num_features) {
					
    if (argc != 2) {
        printf("Need exactly 1 argument, the input file. Usage:\n");
        printf("\t%s <input_file>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
	
    char *filename = argv[1];
	
    printf("Reading data from %s.\n", filename);
	
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Could not open file %s.\n", filename);
        exit(EXIT_FAILURE);
    }

    size_t read;

    read = fread(num_points, sizeof(*num_points), 1, file);
    if (read != 1) {
        printf("Could not read num_points data from %s.\n", filename);
        exit(EXIT_FAILURE);
    }

    read = fread(num_classes, sizeof(*num_classes), 1, file);
    if (read != 1) {
        printf("Could not read num_classes data from %s.\n", filename);
        exit(EXIT_FAILURE);
    }

    read = fread(num_new_points, sizeof(*num_new_points), 1, file);
    if (read != 1) {
        printf("Could not read num_new_points data from %s.\n", filename);
        exit(EXIT_FAILURE);
    }

    read = fread(k, sizeof(*k), 1, file);
    if (read != 1) {
        printf("Could not read k data from %s.\n", filename);
        exit(EXIT_FAILURE);
    }

    *known_points = malloc(*num_points * sizeof(**known_points));
    read = fread(*known_points, sizeof(**known_points), *num_points, file);
    if (read != (size_t) *num_points) {
        printf("Could not read known_points data from %s.\n", filename);
        exit(EXIT_FAILURE);
    }

    *new_points = malloc(*num_new_points * sizeof(**new_points));
    read = fread(*new_points, sizeof(**new_points), *num_new_points, file);
    if (read != (size_t) *num_new_points) {
        printf("Could not read new_points data from %s.\n", filename);
        exit(EXIT_FAILURE);
    }

    *key = malloc(*num_new_points * sizeof(**key));
    read = fread(*key, sizeof(**key), *num_new_points, file);
    if (read != (size_t) *num_new_points) {
        printf("Could not read key data from %s.\n", filename);
        exit(EXIT_FAILURE);
    }

    printf("%d-NN, %d classes, %d instances, %d features, %d points to classify\n", *k, *num_classes, *num_points,
           num_features, *num_new_points);

    fclose(file);
}

void generate_random_data(int argc, char **argv,
                          int *num_points, int *num_classes, int *num_new_points, int *k,
                          Point **known_points, Point **new_points,
                          char **key, int num_features) {
							  
    if (argc != 2) {
        printf("Need exactly 1 argument, the dataset random version. Usage:\n");
        printf("\t%s <d1 | d2 | d3>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
	
    char * dup = strdup(argv[1]);

    if (strcmp(dup, "d1") == 0) {
		
        *k = 15;
        *num_classes = 8;
        *num_points = 10000;
        *num_new_points = 100;

    } else if (strcmp(dup, "d2") == 0) {
		
        *k = 15;
        *num_classes = 8;
        *num_points = 250000;
        *num_new_points = 100;

    } else if (strcmp(dup, "d3") == 0) {
		
        *k = 15;
        *num_classes = 8;
        *num_points = 1000000;
		*num_new_points = 100;
		
    } else {
        printf("Could not recognize dataset version %s.\n", dup);
        exit(EXIT_FAILURE);
    }
	
	printf("Generating data for %s.\n", dup);
	free(dup);
	
    *key = NULL;
	
	*known_points = malloc(*num_points * sizeof(**known_points));
	initialize_known_points(*num_points, *known_points, *num_classes, num_features);
	
	*new_points = malloc(*num_new_points * sizeof(**new_points));
	initialize_new_points(*num_new_points, *new_points, num_features);
	
	printf("%d-NN, %d classes, %d instances, %d features, %d points to classify\n", *k, *num_classes, *num_points, num_features, *num_new_points);
}
