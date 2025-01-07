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

void copy_k_nearest(BestPoint *dist_points, BestPoint *best_points, int k) {
    for(int i = 0; i < k; i++) {   // we only need the top k minimum distances
       best_points[i].classification_id = dist_points[i].classification_id;
       best_points[i].distance = dist_points[i].distance;
    }
}

// CUDA VERSION
void swap_points(BestPoint *a, BestPoint *b) {
    BestPoint temp = *a;
    *a = *b;
    *b = temp;
}

// Partition function
int partition(BestPoint arr[], int low, int high) {
    DATA_TYPE pivot = arr[high].distance;
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j].distance < pivot) {
            i++;
            swap_points(&arr[i], &arr[j]);
        }
    }
    swap_points(&arr[i + 1], &arr[high]);
    return i + 1;
}

// Partial quicksort to get the first k elements sorted
void select_k_nearest_quick(BestPoint *arr, int low, int high, int k) {
    if (low < high) {
        int pi = partition(arr, low, high); 

        if (pi > k) {
            select_k_nearest_quick(arr, low, pi - 1, k);
        }
        else if (pi < k) {
            select_k_nearest_quick(arr, low, pi - 1, k);
            select_k_nearest_quick(arr, pi + 1, high, k);
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
    
    Point *d_points;
    Point *d_newpoint;
    BestPoint* d_distances;
    
    
    // Allocate GPU memory
    HANDLE_ERROR(cudaMalloc((void**)&d_points, num_points * sizeof(Point)));
    HANDLE_ERROR(cudaMalloc((void**)&d_newpoint, sizeof(Point)));
    HANDLE_ERROR(cudaMalloc((void**)&d_distances, num_points * sizeof(BestPoint)));

    HANDLE_ERROR(cudaMemcpy(d_points, known_points, num_points * sizeof(Point), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_newpoint, new_point, sizeof(Point), cudaMemcpyHostToDevice));
    
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

    select_k_nearest_quick(distances, 0, num_points-1, k-1);

    copy_k_nearest(distances, kpoints, k);

    CLASS_ID_TYPE classID = plurality_voting(k, kpoints, num_classes);

    free(kpoints);
    free(distances);

	return classID;
}