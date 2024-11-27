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


#ifndef HASHTABLE_HASHTABLE_H
#define HASHTABLE_HASHTABLE_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct {

    void *data;
    char *key;
    int  time; // Entry time
    int  access_time; // Last access to data
} HashEntry;

typedef struct {

    HashEntry **entries;
    unsigned int max_entries;
    unsigned int current_entries;
    unsigned int current_time;
    unsigned int last;         // Pos of the last recently used
} HashTable;


HashTable *ht_init(unsigned int max_entries);

HashTable *ht_destroy(HashTable *ht);

HashTable *ht_free_and_destroy(HashTable *ht, void (*free_func)(void *));

void ht_put(HashTable *ht, const char *key, void *data);

void *ht_get(HashTable *ht, const char *key);

void *ht_remove(HashTable *ht, const char *key);

/**
 * The callback function is passed a string with the key and a void* with the data.
 */
void ht_foreach(HashTable *ht, void (*fe_func)(const char *key, void *data));

size_t ht_compute_hash(const char *key);

#endif //HASHTABLE_HASHTABLE_H
