#ifndef KNN_PARAMS_H //KNN_PARAMS_H
#define KNN_PARAMS_H

#include <float.h>

#define NUM_TRAINING_SAMPLES 4336

#define NUM_TESTING_SAMPLES 1082

#define NUM_FEATURES 43

#define NUM_CLASSES 6

#ifndef K
#define K 3 // 3 or 20 for READ = 1 (some authors consider K=sqrt(NUM_TRAINING_INSTANCES) = 65
#endif

#ifndef DT
#define DT 1 // 1: double; 2: float; 3: not used for now
#endif

#if DT == 1	//double
	#define DATA_TYPE double
	#define MAX_FP_VAL DBL_MAX
	#define MIN_FP_VAL -DBL_MAX
#elif DT == 2 //float
	#define DATA_TYPE float
	#define MAX_FP_VAL FLT_MAX
	#define MIN_FP_VAL -FLT_MAX
#endif

#if NUM_CLASSES > 128
	#define CLASS_ID_TYPE short // consider 0..32767 classes and -1 for unknown
#else
	#define CLASS_ID_TYPE char // consider 0..127 classes and -1 for unknown
#endif


#endif //KNN_PARAMS_H