/*
 * Represents a patient's record and the actions to read it from a file.
 */
#ifndef PATIENT_H
#define PATIENT_H

#include <stdio.h>

// Data structures
enum Sex {Female, Male};

struct Date {
    int day, month, year;
};

struct Patient {
    char *name;
    enum Sex sex;
    struct Date birthday;
    int num_conditions, num_medications;
    char **conditions, **medications;
};

// Public functions
struct Patient ReadPatient(FILE *patient_file);
struct Patient *ReadAllPatients(const char *patient_data_filename, size_t *num_patients);

// Internal functions
int ReadBirthday(FILE *patient_file, struct Date *birthday);
int ReadSex(FILE *patient_file, enum Sex *sex);

#endif  // PATIENT_H