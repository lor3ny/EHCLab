/**
*	k-NN
*
*	Versions
*	- v0.1, December 2016
*	- v0.2, November 2019
*	- v0.5, November 2021
*   - v0.6, October 2023
*	- v0.7, October 2024
*
*	by Jo�o MP Cardoso
*	Email: jmpc@fe.up.pt
*
*	SPeCS, FEUP.DEI, University of Porto, Portugal
*/

#include <math.h>
#include <stdlib.h>

extern "C"{

#include "knn.h"

}


// CUDA VERSION
void copy_k_nearest(BestPoint *dist_points, BestPoint *best_points, int k) {

    for(int i = 0; i < k; i++) {   // we only need the top k minimum distances
       best_points[i].classification_id = dist_points[i].classification_id;
       best_points[i].distance = dist_points[i].distance;
    }

}

void select_k_nearest(BestPoint *dist_points, int num_points, int k) {

    DATA_TYPE min_distance, distance_i;
    CLASS_ID_TYPE class_id_1;
    int index;


    for(int i = 0; i < k; i++) {  // we only need the top k minimum distances
		min_distance = dist_points[i].distance;
		index = i;

		for(int j = i+1; j < num_points; j++) {
            if(dist_points[j].distance < min_distance) {
                min_distance = dist_points[j].distance;
                index = j;
            }
      }
      if(index != i) { //swap
         distance_i = dist_points[index].distance;
         class_id_1 = dist_points[index].classification_id;

         dist_points[index].distance = dist_points[i].distance;
         dist_points[index].classification_id = dist_points[i].classification_id;

         dist_points[i].distance = distance_i;
         dist_points[i].classification_id = class_id_1;
      }
    }
}

CLASS_ID_TYPE plurality_voting(int k, BestPoint *best_points, int num_classes) {

	//unsigned CLASS_ID_TYPE *histogram = (unsigned CLASS_ID_TYPE *) calloc(NUM_CLASSES, sizeof(CLASS_ID_TYPE)) ;
    unsigned CLASS_ID_TYPE *histogram = (unsigned CLASS_ID_TYPE*) malloc(sizeof(unsigned CLASS_ID_TYPE) * num_classes);  // maximum is the value of k

    //initialize the histogram
    for (int i = 0; i < num_classes; i++) {
        histogram[i] = 0;
    }

    // build the histogram
    for (int i = 0; i < k; i++) {
        BestPoint p = best_points[i];
        histogram[(int) p.classification_id] += 1;
    }
	
	CLASS_ID_TYPE classification_id = best_points[0].classification_id;
    CLASS_ID_TYPE max = 1; // maximum is k
    for (int i = 0; i < num_classes; i++) {

        if (histogram[i] > max) {
            max = histogram[i];
            classification_id = (CLASS_ID_TYPE) i;
        }
    }

    free(histogram);

    return classification_id;
}

__global__ void ComputeDistances_CUDA_v1(Point *new_point, Point* points, BestPoint* distances){


    int globalThreadPoint = threadIdx.x + blockIdx.x * blockDim.x;
    DATA_TYPE distance = (DATA_TYPE) 0.0;

    for(int i = 0; i<NUM_FEATURES; i++){
        DATA_TYPE diff = (DATA_TYPE) new_point->features[i] - (DATA_TYPE) points[globalThreadPoint].features[i];
        distance += diff*diff;
    }

    distances[globalThreadPoint].classification_id = points[globalThreadPoint].classification_id;
    distances[globalThreadPoint].distance = distance;

    //printf("Point: %d Diff %.6f\n", globalThreadPoint, distances[globalThreadPoint].distance);
}


__global__ void ComputeDistances_CUDA_v2(Point *new_point, Point* points, BestPoint* distances){


    int point_id = blockIdx.x;
    int feature_id = threadIdx.x;

    // Each thread computes one feature's squared difference
    if (feature_id < NUM_FEATURES) {
        DATA_TYPE diff = (DATA_TYPE) new_point->features[feature_id] - (DATA_TYPE) points[point_id].features[feature_id];
        atomicAdd(&distances[point_id].distance, diff * diff);
    }

    // Thread 0 writes the classification ID
    if (threadIdx.x == 0) {
        distances[point_id].classification_id = points[point_id].classification_id;
    }

    //printf("Point: %d Diff %.6f\n", globalThreadPoint, distances[globalThreadPoint].distance);
}

extern "C" 
CLASS_ID_TYPE knn_classifyinstance(Point *new_point, const int k, const int num_classes, Point *known_points, const int num_points, const int num_features) {

	// Allocate CPU memory
    BestPoint* kpoints = (BestPoint*) malloc(sizeof(BestPoint) * k); // Array with the k nearest points to the Point to classify
    BestPoint* distances = (BestPoint*) malloc(sizeof(BestPoint) * num_points);
    
    Point *d_kpoints;
    Point *d_points;
    Point *d_newpoint;
    BestPoint* d_distances;
    
    
    // Allocate GPU memory
    HANDLE_ERROR(cudaMalloc((void**)&d_points, num_points * sizeof(Point)));
    HANDLE_ERROR(cudaMalloc((void**)&d_newpoint, sizeof(Point)));
    HANDLE_ERROR(cudaMalloc((void**)&d_distances, num_points * sizeof(BestPoint)));
    //HANDLE_ERROR(cudaMalloc((void**)&d_kpoints, k * sizeof(Point)));

    HANDLE_ERROR(cudaMemcpy(d_points, known_points, num_points * sizeof(Point), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_newpoint, new_point, sizeof(Point), cudaMemcpyHostToDevice));
    //HANDLE_ERROR(cudaMemcpy(d_distances, distances, num_points * sizeof(BestPoint), cudaMemcpyHostToDevice));
    
    #if CUDA_VERSION == 1
        int threads_per_block = 256; 
        int num_blocks = (num_points + threads_per_block - 1) / threads_per_block;
        ComputeDistances_CUDA_v1<<<num_blocks, threads_per_block>>>(d_newpoint, d_points, d_distances);
    #else
        int threads_per_block = num_features; 
        int num_blocks = num_points;
        ComputeDistances_CUDA_v2<<<num_blocks, threads_per_block>>>(d_newpoint, d_points, d_distances);
    #endif

    cudaDeviceSynchronize();

    HANDLE_ERROR(cudaMemcpy(distances, d_distances, num_points * sizeof(BestPoint), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(d_points));
    HANDLE_ERROR(cudaFree(d_newpoint));
    HANDLE_ERROR(cudaFree(d_distances));
    //HANDLE_ERROR(cudaFree(d_kpoints));

    /*
    for (int i = 0; i < num_points; i++) {
		printf("IDdefault = %d | IDcuda = %d | dist_def = %.6f | dist_cuda = %.6f\n", dist_points[i].classification_id, distances[i].classification_id, dist_points[i].distance, distances[i].distance);
    }
    */

    select_k_nearest(distances, num_points, k);

    copy_k_nearest(distances, kpoints, k);

    CLASS_ID_TYPE classID = plurality_voting(k, kpoints, num_classes);

    free(kpoints);
    free(distances);

	return classID;
}