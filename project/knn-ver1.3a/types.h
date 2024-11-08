
#ifndef KNN_TYPES_H
#define KNN_TYPES_H

#include "params.h"

typedef struct {
    DATA_TYPE features[NUM_FEATURES];
    CLASS_ID_TYPE classification_id;
} Point;

typedef struct {
    CLASS_ID_TYPE classification_id;
    DATA_TYPE distance;
} BestPoint;

#endif //KNN_TYPES_H
