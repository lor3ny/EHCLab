#ifndef KNN_UTILS_H
#define KNN_UTILS_H

#include <math.h>

#include "params.h"
#include "types.h"

void verify_results(int num_new_points, const Point *new_points, const CLASS_ID_TYPE *key);

void initialize_known_points(int num_points, Point *known_points, int num_classes, 
						int num_features);

void initialize_new_points(int num_new_points, Point *new_points, int num_features);

void show_points(int num_points, Point *points, int num_features);
	
void show_point(Point point, int num_features);

void minmax(DATA_TYPE *min, DATA_TYPE *max, int num_points, Point *known_points, 
			int num_features);
			
void minmax_normalize(DATA_TYPE *min, DATA_TYPE *max, int num_points, Point *points, 
					int num_features);
					
void minmax_normalize_point(DATA_TYPE *min, DATA_TYPE *max, Point *point, int num_features);

#endif
