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

#include "knn.h"


// CUDA VERSION


__global__ void ComputeDistances_GPU(Point *new_point, CUDA_Point* points, float* distances){

    //int globalThreadIdx = threadIdx.x + blockIdx.x * blockDim.x;
    
    int point_id = blockIdx.x;
    int feat_id = threadIdx.x;

    float diff = (float) new_point->features[feat_id] - (float) points[point_id].features[feat_id];
    diff = diff*diff;
    atomicAdd(&points[point_id].distance, diff); //distances[point_id] += diff;

    // Non sto usuando la versione con distances, capire se è necessario
}

CLASS_ID_TYPE knn_classifyinstance_CUDA(Point *new_point, const int k, const int num_classes, CUDA_Point *known_points, int num_points, int num_features) {

	//BestPoint *best_points = (BestPoint *) calloc(k, sizeof(BestPoint)) ;
    BestPoint* k_points = (BestPoint*) malloc(sizeof(BestPoint) * k); // Array with the k nearest points to the Point to classify

    Point *d_kpoints;
    CUDA_Point *d_points;
    float* d_distances;

    
    // Allocate GPU memory
    cudaMalloc((void**)&d_distances, num_points * sizeof(float));
    cudaMalloc((void**)&d_kpoints, k * sizeof(Point));
    cudaMalloc((void**)&d_points, num_points * sizeof(Point));

    int num_blocks = num_points;
    int num_threads = num_features;
    ComputeDistances_GPU<<<num_blocks, num_threads>>>(new_point, d_points, d_distances);

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

    cudaFree(d_distances);
    cudaFree(d_kpoints);
    cudaFree(d_points);

	return classID;
}