/*
 * Defines utility functions for working with in-file lists
 */
#ifndef LIST_H
#define LIST_H

#include <stdio.h>

/*
 * Read a list of conditions or medications from the provided stream.
 * The first line read from the buffer should indicate the number of items to read.
 * Each item is delimited by a newline.
 * Both the header line and the item lines may be prefixed with any number of
 * whitespace characters which will be discarded.
 * *medications_list will be allocated to store the read items. This value and the
 * strings it contains should be freed even if this method exits with an error.
 * Returns: the number of items read, or -1 if there was an error.
 */
ssize_t ReadListItems(FILE *list_file, char ***list);

#endif  // LIST_H