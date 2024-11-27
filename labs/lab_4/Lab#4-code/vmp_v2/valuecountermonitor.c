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


#include <stdbool.h>
#include "valuecountermonitor.h"

/**
 * The file used to write the JSON report.
 */
static FILE *f;

/**
 * The boolean to control the printing of the JSON counts array.
 */
static bool first;

static void my_uint_free(void *data);

static void print_table(const char *key, void *data);

static void print_json(const char *key, void *data);

void vcm_inc(VCM *vcm, double val) {
    
    if (vcm != NULL) {

        vcm->total_occ++;

        if((vcm->min <= val && vcm->max >= val && vcm->range == 1) || (vcm->range == 0) )
        {
            int aux, aux_pos;
            int * aux_data;

            // Create key from value
            char key[50];
            snprintf(key, 50, "%.*f", vcm->precision, val);

            // Get value and increment or create new
            void *data = NULL;

            // If the table has entries try to get the key value
            if (vcm->table->current_entries != 0) 
            {
                data = ht_get(vcm->table, key);
            }

            unsigned int *new_data = (unsigned int *) malloc(sizeof(unsigned int)); 

            // If the value doesnt exist yet verify if there is space left
            if (data == NULL) 
            {
                vcm->total_values ++;

                // There is space
                if (vcm->table->current_entries != vcm->table->max_entries) {

                    *new_data = 1;
                    ht_put(vcm->table, key, new_data);
                }
                // If table is full -> Substitution policies
                else
                {
                    vcm->total_subs++;

                    if(vcm->subsPolicy == FIFO)
                    {
                        *new_data = 1;

                        int i;

                        aux = vcm->table->entries[0]->time;
                        aux_pos = 0;

                        // Remove the first in 
                        for(i = 0; i < vcm->table->current_entries ; i ++)
                        {
                            if(vcm->table->entries[i]->time < aux)
                            {
                                aux = vcm->table->entries[i]->time;
                                aux_pos = i;
                            }
                        }

                        ht_remove(vcm->table,vcm->table->entries[aux_pos]->key);

                        // Put the new

                        ht_put(vcm->table, key, new_data);
                    
                    }
                    else if(vcm->subsPolicy == LRU)
                    {
                        *new_data = 1;

                        int i;

                        aux = vcm->table->entries[0]->access_time;
                        aux_pos = 0;

                        // Search the less frquently used
                        for(i = 0; i < vcm->table->current_entries ; i ++)
                        {
                            if(vcm->table->entries[i]->access_time < aux)
                            {
                                aux = vcm->table->entries[i]->access_time;
                                aux_pos = i;
                            }
                        }

                        ht_remove(vcm->table,vcm->table->entries[aux_pos]->key);

                        // Put the new

                        ht_put(vcm->table, key, new_data);
                    }
                    else if(vcm->subsPolicy == LFU)
                    {
                        *new_data = 1;

                        int i;

                        aux_data = (int *)vcm->table->entries[0]->data;

                        // Search the less frquently used
                        for(i = 0; i < vcm->table->current_entries ; i ++)
                        {
                            if(*(int *)vcm->table->entries[i]->data < *aux_data)
                            {
                                aux_data = (int *)vcm->table->entries[i]->data;
                                aux_pos = i;
                            }
                        }

                        ht_remove(vcm->table,vcm->table->entries[aux_pos]->key);

                        // Put the new

                        ht_put(vcm->table, key, new_data);
                    }
                    else if(vcm->subsPolicy == MRU)
                    {
                        *new_data = 1;
                    
                        ht_remove(vcm->table,vcm->table->entries[vcm->table->last]->key);
                        ht_put(vcm->table, key, new_data);
                    }
                    else if (vcm->subsPolicy == RAND)
                    {
                        *new_data = 1;
                        int rand_pos =  rand() % (vcm->table->current_entries);
                        ht_remove(vcm->table,vcm->table->entries[rand_pos]->key);
                        ht_put(vcm->table, key, new_data);
                    }
                }

            // In case the key value already exists increments 1
            } else {

                unsigned int *uip = (unsigned int *) data;
                (*uip)++;

                size_t hash = ht_compute_hash(key);
                size_t pos = hash % vcm->table->max_entries;

                vcm->table->last = pos;

                vcm->table->entries[pos]->access_time = vcm->table->current_time;
                vcm->table->current_time++;
            }
        }
    }
}

VCM *vcm_init(const char *name, unsigned int max_entries, unsigned int precision, int policy, int min, int max, int range) {

    VCM *vcm = (VCM *) malloc(sizeof(VCM));

    size_t length = strlen(name) + 1;
    vcm->name = (char *) malloc(length);
    strcpy(vcm->name, name);

    vcm->table = ht_init(max_entries);

    vcm->precision = precision;

    vcm->subsPolicy = policy;

    vcm->total_occ    = 0;
    vcm->total_subs   = 0;
    vcm->total_values = 0;

    vcm->min = min;
    vcm->max = max;
    vcm->range = range;

    return vcm;
}

VCM *vcm_destroy(VCM *vcm) {

    free(vcm->name);

    ht_free_and_destroy(vcm->table, my_uint_free);

    free(vcm);

    return NULL;
}

void vcm_print(VCM *vcm) {

    printf("Table '%s', %u elements\n\n", vcm->name, vcm->table->current_entries);
    ht_foreach(vcm->table, print_table);

    printf("Total values = %d \nTotal occurrences = %d \nTotal substitutions = %d \n", vcm->total_values, vcm->total_occ, vcm->total_subs);
}

void vcm_to_json(VCM *vcm, const char *filename) {

    f = fopen(filename, "w");
    if (f == NULL) {
        printf("Error opening file: %s\n", filename);
        return;
    }

    first = true;

    fprintf(f, "{");

    fprintf(f, "\"name\" : \"%s\",", vcm->name);
    fprintf(f, "\"elements\" : \"%u\",", vcm->table->current_entries);

    fprintf(f, "\"counts\" : [");
    ht_foreach(vcm->table, print_json);
    fprintf(f, "]");

    fprintf(f, "}");

    fclose(f);
    f = NULL;
}


static void my_uint_free(void *data) {

    unsigned int *uip = (unsigned int *) data;
    free(uip);
}


static void print_table(const char *key, void *data) {

    unsigned int *uip = (unsigned int *) data;
    printf("%s : %u\n", key, *uip);
}


static void print_json(const char *key, void *data) {

    unsigned int *uip = (unsigned int *) data;

    if (first) {

        fprintf(f, "{\"%s\" : \"%u\"}", key, *uip);
        first = false;
    } else {

        fprintf(f, ",{\"%s\" : \"%u\"}", key, *uip);
    }
}
