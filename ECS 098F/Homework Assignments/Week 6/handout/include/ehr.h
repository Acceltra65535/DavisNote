/*
 * Program logic for the EHR
 */
#ifndef EHR_H
#define EHR_H

#include <stddef.h>

struct Patient;

struct Patient *FindPatientByName(char *name);
int EhrMain(const char *patient_data_filename);

#endif  //EHR_H