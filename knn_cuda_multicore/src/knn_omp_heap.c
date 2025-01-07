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

#include "knn.h"

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

    #pragma omp simd
    for (int i = 0; i < k; i++) {
        result[i] = heap[i];
    }

    // Free the allocated memory
    free(heap);
}


/*
* Main kNN function.
* It calculates the distances and calculates the nearest k points.
*/
void get_k_NN(Point *new_point, Point *known_points, int num_points,
	BestPoint *best_points, int k,  int num_features) {
     
    BestPoint dist_points[num_points];

    // calculate the Euclidean distance between the Point to classify and each Point in the
    // training dataset (knowledge base)
    #pragma omp parallel for
    for (int i = 0; i < num_points; i++) {
        DATA_TYPE distance = (DATA_TYPE) 0.0;

        // calculate the Euclidean distance
        #pragma omp simd
        for (int j = 0; j < num_features; j++) {
            DATA_TYPE diff = (DATA_TYPE) new_point->features[j] - (DATA_TYPE) known_points[i].features[j];
            distance += diff * diff;
        }
		
		//distance = sqrt(distance);
		
        dist_points[i].classification_id = known_points[i].classification_id;
        dist_points[i].distance = distance;
    }

    find_top_k(dist_points, num_points, k, best_points);
}

/*
*	Classify using the k nearest neighbors identified by the get_k_NN
*	function. The classification uses plurality voting.
*
*	Note: it assumes that classes are identified from 0 to
*	num_classes - 1.
*/
CLASS_ID_TYPE plurality_voting(int k, BestPoint *best_points, int num_classes) {

	//unsigned CLASS_ID_TYPE *histogram = (unsigned CLASS_ID_TYPE *) calloc(NUM_CLASSES, sizeof(CLASS_ID_TYPE)) ;
    unsigned CLASS_ID_TYPE histogram[num_classes];  // maximum is the value of k

    //initialize the histogram
    #pragma omp parallel for
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

    return classification_id;
}

/*
* Classify a given Point (instance).
* It returns the classified class ID.
*/
CLASS_ID_TYPE knn_classifyinstance(Point *new_point, int k, int num_classes, Point *known_points, int num_points, int num_features) {

	//BestPoint *best_points = (BestPoint *) calloc(k, sizeof(BestPoint)) ;
     BestPoint best_points[k]; // Array with the k nearest points to the Point to classify

    // calculate the distances of the new point to each of the known points and get
    // the k nearest points
    get_k_NN(new_point, known_points, num_points, best_points, k, num_features);

	// use plurality voting to return the class inferred for the new point
	CLASS_ID_TYPE classID = plurality_voting(k, best_points, num_classes);

	// content of the k best
	//for (int i = 0; i < k; i++) {
		//printf("ID = %d | distance = %e\n",best_points[i].classification_id, best_points[i].distance);
    //}
	return classID;
}