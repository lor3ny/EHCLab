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


__global__ void ComputeDistances_CUDA(Point *new_point, Point* points, float* distances){

    //int globalThreadIdx = threadIdx.x + blockIdx.x * blockDim.x;
    
    int point_id = blockIdx.x;
    int feat_id = threadIdx.x;

    float diff = (float) new_point->features[feat_id] - (float) points[point_id].features[feat_id];
    diff = diff*diff;
    atomicAdd(&distances[feat_id], diff); //distances[point_id] += diff;

    // Non sto usuando la versione con distances, capire se è necessario
}

extern "C" 
CLASS_ID_TYPE knn_classifyinstance_CUDA(Point *new_point, int k, int num_classes, Point *known_points, int num_points, const int num_features) {

	//BestPoint *best_points = (BestPoint *) calloc(k, sizeof(BestPoint)) ;
    BestPoint* k_points = (BestPoint*) malloc(sizeof(BestPoint) * k); // Array with the k nearest points to the Point to classify

    Point *d_kpoints;
    Point *d_points;
    float* d_distances;
    float* distances = (float*) malloc(sizeof(float)*num_points);

    
    // Allocate GPU memory
    cudaMalloc((void**)&d_distances, num_points * sizeof(float));
    cudaMalloc((void**)&d_kpoints, k * sizeof(Point));
    cudaMalloc((void**)&d_points, num_points * sizeof(Point));

    cudaMemcpy(d_points, known_points, num_points * sizeof(Point), cudaMemcpyHostToDevice);

    int num_blocks = num_points;
    int num_threads = num_features;
    ComputeDistances_CUDA<<<num_blocks, num_threads>>>(new_point, d_points, d_distances);

    //MergeDistances_GPU Probablimente non serve se faccio le atomic add
    // A quel punto sostituire distances[point_id] con points[i]
 
    //SelectKNN<<<1, 256>>>(d_points, d_kpoints);

	// use plurality voting to return the class inferred for the new point
	//CLASS_ID_TYPE classID = plurality_voting(k, best_points, num_classes);

    CLASS_ID_TYPE classID = 1;

	// content of the k best
	//for (int i = 0; i < k; i++) {
		//printf("ID = %d | distance = %e\n",best_points[i].classification_id, best_points[i].distance);
    //}

    cudaMemcpy(distances, d_distances, num_points * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_points; i++) {
		printf("ID = %d | distance = %e\n", i, distances[i]);
    }

    cudaFree(d_distances);
    cudaFree(d_kpoints);
    cudaFree(d_points);

    //CLASS_ID_TYPE classID = plurality_voting(k, best_points, num_classes);

	// content of the k best
	//for (int i = 0; i < k; i++) {
		//printf("ID = %d | distance = %e\n",best_points[i].classification_id, best_points[i].distance);
    //}

	return classID;
}