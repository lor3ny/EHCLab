/**
*            VMP: Value Counter Monitor
*
*            Versions 
*            - v1, May 2016; by Pedro Pinto
*            - v2, June 2023; by Vitória Correia 
*            
*            v1: library for runtime monitoring range values of program variables
*            v2: library for runtime monitoring values of program variables and with 
*                           implementation of substitution policies
*
*            SPeCS, FEUP.DEI, University of Porto, Portugal
*/


#ifndef VALUECOUNTERMONITOR_VALUECOUNTERMONITOR_H
#define VALUECOUNTERMONITOR_VALUECOUNTERMONITOR_H

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include "hashtable.h"

#define FIFO 1  // First in First Out
#define LRU  2  // Least Recently Used
#define LFU  3  // Least Frequently Used
#define MRU  4  // Most Recently Used
#define RAND 5  // Random

typedef struct {

    char* name;
    HashTable* table;
    unsigned int precision;
    int subsPolicy;
    int total_values; // Total different values for parameter
    int total_occ;    // Total ocurrences of all values
    int total_subs;   // Total of substitutions in hashtable
    int min;          // min value in table
    int max;          // mÃ¡x Value in table
    int range;        // 1 if range enable, 0 no limit

} VCM;

/**
 * We keep a copy of the name string.
 */
VCM *vcm_init(const char *name, unsigned int max_entries, unsigned int precision, int policy, int min, int max, int range);
VCM* vcm_destroy(VCM* vcm);
void vcm_inc(VCM *vcm, double val);
void vcm_print(VCM* vcm);
void vcm_to_json(VCM* vcm, const char* filename);


#endif //VALUECOUNTERMONITOR_VALUECOUNTERMONITOR_H
