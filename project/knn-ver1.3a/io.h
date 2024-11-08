#ifndef KNN_IO_H
#define KNN_IO_H

#include "utils.h"

/**
* Read data from .dat files.
*/
void read_data(int argc, char **argv,
               int *num_points, int *num_classes, int *num_new_points, int *k,
               Point **known_points, Point **new_points,
               char **key, int num_features);

/**
 * Use only if you can't load data with read_data.
 */
void generate_random_data(int argc, char **argv,
                          int *num_points, int *num_classes, int *num_new_points, int *k,
                          Point **known_points, Point **new_points,
                          char **key, int num_features);
						  
#endif
