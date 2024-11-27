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
#include "hashtable.h"

/**
 * Private table functions.
 */
static size_t ht_get_new_pos(HashTable *table, size_t pos);

size_t ht_compute_hash(const char *key);

static bool ht_check_key(const char *key, const char *type, const char *operation);

static bool ht_check_full(const HashTable *ht, const char *type, const char *operation);

static bool ht_check_empty(const HashTable *ht, const char *type, const char *operation);

static bool ht_check_entries(unsigned int entries, const char *type);

static bool ht_check_duplicate(HashTable *ht, const char *key, const char *type, const char *operation);

/**
 * Private entry functions.
 */
static HashEntry *he_init(const char *key, void *data);

static HashEntry *he_destroy(HashEntry *he);


HashTable *ht_init(unsigned int max_entries) {

    if (!ht_check_entries(max_entries, "ERROR")) {

        return NULL;
    }

    unsigned int i;

    HashTable *ht = (HashTable *) malloc(sizeof(HashTable));

    ht->entries = (HashEntry **) malloc(sizeof(HashEntry *) * max_entries);
    for (i = 0; i < max_entries; i++) {

        ht->entries[i] = NULL;
    }

    ht->max_entries = max_entries;
    ht->current_entries = 0;
    ht->current_time = 0;

    return ht;
}


/**
 * Keeps a copy of the key. Does not keep a copy of the data.
 *
 * If this function is called several times with the same key, only the first will be performed.
 */
void ht_put(HashTable *ht, const char *key, void *data) {

    if (!ht_check_key(key, "WARN", "put()") ||
        !ht_check_full(ht, "WARN", "put()") ||
        !ht_check_duplicate(ht, key, "WARN", "put()")) {
        return;
    }

    /* Get the position of this record. */
    size_t hash = ht_compute_hash(key);
    size_t pos = hash % ht->max_entries;

    /* Check if there is a collision and get a valid position if needed. */
    if (ht->entries[pos] != NULL) {

        pos = ht_get_new_pos(ht, pos);
    }

    /* Insert the entry. */
    HashEntry *entry = he_init(key, data);
    ht->entries[pos] = entry;
    ht->entries[pos]->time = ht->current_time;
    ht->current_entries++;
    ht->current_time++;
    ht->last = pos;
    
}


/**
 * Returns a pointer to the data or NULL if no mapping with the given key is found.
 */
void *ht_get(HashTable *ht, const char *key) {

    if (!ht_check_key(key, "ERROR", "get()") ||
        !ht_check_empty(ht, "ERROR", "get()")) {
        return NULL;
    }

    /* Counter use to avoid infinite looping around the table if it is full and we ahve a non-mapped key. */
    unsigned int counter = 0;

    /* Using the key, get the hash and the would-be position of the record. */
    size_t hash = ht_compute_hash(key);
    size_t pos = hash % ht->max_entries;

    while (ht->entries[pos] != NULL
           && strcmp(key, ht->entries[pos]->key) != 0
           && counter <= ht->max_entries) {

        pos = (pos + 1) % ht->max_entries;
        counter++;
    }

    /* Return either the data, or null if no entry with that key was found. */
    if (ht->entries[pos] == NULL || counter > ht->max_entries) {
        return NULL;
    } else {
        return ht->entries[pos]->data;
    }
}

/**
 * To get the pointer to the data being removed from the table, call like this:
 *      void* data = ht_remove(ht, key);
 *
 * After removal, we not longer keep a pointer to the data, so it becomes your responsability to keep track of your pointers and free them.
 */
void *ht_remove(HashTable *ht, const char *key) {

    if (!ht_check_key(key, "WARN", "remove()") ||
        !ht_check_empty(ht, "WARN", "remove()")) {
        return NULL;
    }

    /* Get the position of this record. */
    size_t hash = ht_compute_hash(key);
    size_t pos = hash % ht->max_entries;


    while (ht->entries[pos] != NULL
           && strcmp(key, ht->entries[pos]->key) != 0) {

        pos = (pos + 1) % ht->max_entries;
    }

    /* If we find a non-NULL entry, we return the data, free the hash_entry and set the position to NULL. */
    if (ht->entries[pos] != NULL) {

        void *data = ht->entries[pos]->data;

        he_destroy(ht->entries[pos]);
        ht->entries[pos] = NULL;
        ht->current_entries--;

        return data;
    }

    return NULL;
}

/**
 * In order to NULL the pointer automatically, call like this:
 *      ht = ht_destroy(ht);
 */
HashTable *ht_destroy(HashTable *ht) {

    return ht_free_and_destroy(ht, NULL);
}


/**
 * In order to NULL the pointer automatically, call like this:
 *      ht = ht_free_and_destroy(ht, func);
 */
HashTable *ht_free_and_destroy(HashTable *ht, void (*free_func)(void *)) {

    if (ht != NULL) {

        unsigned int i;
        for (i = 0; i < ht->max_entries; i++) {

            HashEntry *entry = ht->entries[i];
            if (entry != NULL) {

                if (free_func != NULL) {

                    free_func(entry->data);
                }

                he_destroy(entry);
            }
        }

        free(ht->entries);
        free(ht);
    }

    return NULL;
}

void ht_foreach(HashTable *ht, void (*fe_func)(const char *key, void *data)) {

    if (ht != NULL) {

        unsigned int i;
        for (i = 0; i < ht->max_entries; i++) {

            HashEntry *entry = ht->entries[i];
            if (entry != NULL) {

                if (fe_func != NULL) {

                    fe_func(entry->key, entry->data);
                }
            }
        }
    }
}


/*
 * Name         : ht_hash
 *
 * Synopsis     : static unsigned ht_compute_hash(const char* key)
 *
 * Arguments    : const char * key : The key of the data.
 *
 * Description  : Given a key of a piece of data, calculates its hash. Uses the One-at-a-Time hashing algorithm.
 *
 * Returns      : unsigned int : The hash of the string.
 */
size_t ht_compute_hash(const char *key) {

    size_t i;
    size_t len = strlen(key);
    size_t h = 0;

    for (i = 0; i < len; i++) {

        h += key[i];
        h += (h << 10);
        h ^= (h >> 6);
    }

    h += (h << 3);
    h ^= (h >> 11);
    h += (h << 15);

    return h;
}

/**
 * Uses Open Addressing with Linear Probing.
 */
static size_t ht_get_new_pos(HashTable *table, size_t pos) {

    do {

        pos++;
        if (pos == table->max_entries) {

            pos = 0;
        }
    } while (table->entries[pos] != NULL);

    return pos;
}

HashEntry *he_init(const char *key, void *data) {

    HashEntry *entry = (HashEntry *) malloc(sizeof(HashEntry));

    size_t length = strlen(key) + 1;
    entry->key = (char *) malloc(length);
    strcpy(entry->key, key);

    entry->data = data;

    return entry;
}

HashEntry *he_destroy(HashEntry *he) {

    if (he != NULL) {

        free(he->key);
        free(he);
    }

    return NULL;
}

static bool ht_check_full(const HashTable *ht, const char *type, const char *operation) {

    if (ht->current_entries == ht->max_entries) {

        fprintf(stderr, "[%s] Cannot %s in HashTable: HashTable is full.\n", type, operation);
        return false;
    }

    return true;
}

static bool ht_check_empty(const HashTable *ht, const char *type, const char *operation) {

    if (ht->current_entries == 0) {

        fprintf(stderr, "[%s] Cannot %s in HashTable: HashTable is empty.\n", type, operation);
        return false;
    }

    return true;
}

static bool ht_check_key(const char *key, const char *type, const char *operation) {

    if (key == NULL) {

        fprintf(stderr, "[%s] Cannot %s in HashTable: the provided key is NULL.\n", type, operation);
        return false;
    }

    return true;
}

static bool ht_check_entries(unsigned int entries, const char *type) {

    if (entries == 0) {

        fprintf(stderr, "[%s] Cannot create HashTable: max entries = 0.\n", type);
        return false;
    }

    return true;
}

static bool ht_check_duplicate(HashTable *ht, const char *key, const char *type, const char *operation) {

    if (ht->current_entries == 0) {

        return true;
    }

    void *test_entry = ht_get(ht, key);

    if (test_entry != NULL) {

        fprintf(stderr, "[%s] Cannot %s in HashTable: duplicate key.\n", type, operation);
        return false;
    }

    return true;
}
