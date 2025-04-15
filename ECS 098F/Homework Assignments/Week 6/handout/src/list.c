#include "list.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

ssize_t ReadListItems(FILE *list_file, char ***list) {
    assert(list != NULL);
    int num_items;
    if(fscanf(list_file, "\n%d", &num_items) != 1) {
        perror("Failed to determine the number of items in a list.");
        return -1; 
    }

    if(num_items == 0) {
        // Clear any whitespace left in the input and exit
        fscanf(list_file, "\n");
        return 0;
    }

    *list = malloc(num_items * sizeof(char*));
    size_t zero = 0;
    for(int i = 0; i < num_items; i++) {
        // Discard any leading whitespace
        fscanf(list_file, "\n");
        char *item = NULL;
        ssize_t length = getline(&item, &zero, list_file);
        if(length == -1) {
            perror("An error occurred while reading list items.");
            return -1;
        }
        // getline will read a newline character if there is a proceeding line.
        // The newline character should not be part of the item, so we remove it.
        item[length - 1] = '\0';

        *list[i] = item;
    }
    return num_items;
}
