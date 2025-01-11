
#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"


/*
* Verify if the classifications equal the original ones stored in key
*/
void verify_results(int num_new_points, const Point *new_points, const CLASS_ID_TYPE *key) {
	
    if (key == NULL) {
        printf("Skipping verification.\n");
        return;
    }

    int passed = 1;
    printf("Verifying results...\n");
    for (int i = 0; i < num_new_points; ++i) {

        CLASS_ID_TYPE classified = new_points[i].classification_id;
        CLASS_ID_TYPE truth = key[i];

        if (classified == truth) {
            printf(" %d %s %d\n", classified, "=", truth);
        } else {
            printf(" %d %s %d\n", classified, "!=", truth);
            passed = 0;
        }

    }

    printf("Verification is complete: ");
    if (passed == 1) {
        printf("Passed!\n");
    } else {
        printf("Failed!\n");
    }
}

/*
* return an integer number from min to max.
*/
static int rand_int(int min, int max) {

    int number = (int) (min + rand() / (RAND_MAX / (max - min + 1) + 1));
    return number;
}

/*
* return a floating-point number from min to max.
*/
static DATA_TYPE rand_double(DATA_TYPE min, DATA_TYPE max) {

    DATA_TYPE number = (DATA_TYPE) (min + rand() / (RAND_MAX / (max - min + 1) + 1));
    return number;
}

/*
* return a floating-point number between 0 and 1
*/
static DATA_TYPE get_rand_feature_value() {

    return rand_double(0, 1);
}

/*
* Initialize points with random values
*/
void initialize_known_points(int num_points, Point *known_points, int num_classes, int num_features) {

    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < num_features; j++) {
            known_points[i].features[j] = (DATA_TYPE) get_rand_feature_value();
        }
        known_points[i].classification_id = (CLASS_ID_TYPE) rand_int(0, num_classes - 1);
    }
}

/*
* Initialize new points with random values.
*/
void initialize_new_points(int num_new_points, Point *new_points, int num_features) {

    for (int i = 0; i < num_new_points; i++) {
        for (int j = 0; j < num_features; j++) {
            new_points[i].features[j] = (DATA_TYPE) get_rand_feature_value();
        }
        new_points[i].classification_id = (CLASS_ID_TYPE) -1;
    }
}

/*
* Show points
*/
void show_points(int num_points, Point *points, int num_features) {

    for (int i = 0; i < num_points; i++) {
		show_point(points[i], num_features);
    }
}

/*
* show the values of a point: features and class.
*/
void show_point(Point point, int num_features) {

    for (int j = 0; j < num_features; j++) {
		if(j == 0)
			printf("%.3f", point.features[j]);
		else
			printf(",%.3f", point.features[j]);
    }
    printf(", class = %d\n", point.classification_id);
}

/*
* Determine the min and max values for each feature for a set of 
* points.
*/ 
void minmax(DATA_TYPE *min, DATA_TYPE *max, int num_points, Point *known_points, int num_features) {

    
	for (int j = 0; j < num_features; j++) {
		min[j] = MAX_FP_VAL;
		max[j] = MIN_FP_VAL;
		//printf("%e, %e\n", MIN_FP_VAL, MAX_FP_VAL);
	}
	
	for (int i = 0; i < num_points; i++) {
		for (int j = 0; j < num_features; j++) {
            if(known_points[i].features[j] < min[j]) 
				min[j] = known_points[i].features[j];
            if(known_points[i].features[j] > max[j]) 
				max[j] = known_points[i].features[j];
        }
    }
	
	/*printf("{");
	for (int j = 0; j < num_features; j++) {
		if(j<num_features-1) printf("%.4f,",min[j]);
		else printf("%.4f",min[j]);
	}
	printf("}\n");
	
	printf("{");
	for (int j = 0; j < num_features; j++) {
		if(j<num_features-1) printf("%.4f,",max[j]);
		else printf("%.4f",max[j]);
	}
	printf("}\n");
	*/
}

/*
* Normalize the features of each point using minmax normalization.
*/
void minmax_normalize(DATA_TYPE *min, DATA_TYPE *max, int num_points, Point *points, int num_features) {

    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < num_features; j++) {
			
			DATA_TYPE nfeature = (DATA_TYPE) ((points[i].features[j] - min[j])/(max[j] - min[j]));
			
			// in case the normalization returns a NaN or INF
			if(isnan(nfeature)) nfeature = (DATA_TYPE) 0.0;
			else if(isinf(nfeature)) nfeature = (DATA_TYPE) 1.0;
			
			points[i].features[j] = nfeature;
		}
		//show_point(points[i], num_features); 
    }
}

/*
* Normalize the features of a single point using minmax normalization.
*/
void minmax_normalize_point(DATA_TYPE *min, DATA_TYPE *max, Point *point, int num_features) {

    for (int j = 0; j < num_features; j++) {
			
		DATA_TYPE nfeature = (DATA_TYPE) ((point->features[j] - min[j])/(max[j] - min[j]));
			
			// in case the normalization returns a NaN or INF
			if(isnan(nfeature)) nfeature = (DATA_TYPE) 0.0;
			else if(isinf(nfeature)) nfeature = (DATA_TYPE) 1.0;
			
			point->features[j] = nfeature;
    }
}

char *custom_strdup(const char *str) {
    if (!str) return NULL;
    size_t len = strlen(str) + 1;
    char *dup = (char *)malloc(len);
    if (dup) {
        memcpy(dup, str, len);
    }
    return dup;
}
