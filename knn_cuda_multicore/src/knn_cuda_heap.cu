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
void swap(BestPoint *a, BestPoint *b) {
    BestPoint temp = *a;
    *a = *b;
    *b = temp;
}

void max_heapify(BestPoint *heap, int heap_size, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < heap_size && heap[left].distance > heap[largest].distance)
        largest = left;

    if (right < heap_size && heap[right].distance > heap[largest].distance)
        largest = right;

    if (largest != i) {
        swap(&heap[i], &heap[largest]);
        max_heapify(heap, heap_size, largest);
    }
}


void build_max_heap(BestPoint *heap, int heap_size) {
    for (int i = (heap_size / 2) - 1; i >= 0; i--) {
        max_heapify(heap, heap_size, i);
    }
}


void find_top_k(BestPoint *arr, int n, int k, BestPoint *result) {
    // Allocate memory for the max heap
    BestPoint *heap = (BestPoint *) malloc(k * sizeof(BestPoint));

    // Initialize the heap with the first k elements
    for (int i = 0; i < k; i++) {
        heap[i] = arr[i];
    }


    build_max_heap(heap, k);


    for (int i = k; i < n; i++) {
        if (arr[i].distance < heap[0].distance) {
            // Replace the root of the heap if the current distance is smaller
            heap[0] = arr[i];
            max_heapify(heap, k, 0);
        }
    }


    for (int i = 0; i < k; i++) {
        result[i] = heap[i];
    }

    // Free the allocated memory
    free(heap);
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

}


__global__ void ComputeDistances_CUDA_v2(Point *new_point, Point* points, BestPoint* distances){

    int point_id = blockIdx.x;
    int feature_id = threadIdx.x;

    // Each thread computes one feature's difference
    if (feature_id < NUM_FEATURES) {
        DATA_TYPE diff = (DATA_TYPE) new_point->features[feature_id] - (DATA_TYPE) points[point_id].features[feature_id];
        atomicAdd(&distances[point_id].distance, diff * diff);
    }

    // Thread 0 writes the classification ID
    if (threadIdx.x == 0) {
        distances[point_id].classification_id = points[point_id].classification_id;
    }

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

    find_top_k(distances, num_points, k, kpoints);

    CLASS_ID_TYPE classID = plurality_voting(k, kpoints, num_classes);

    free(kpoints);
    free(distances);

	return classID;
}