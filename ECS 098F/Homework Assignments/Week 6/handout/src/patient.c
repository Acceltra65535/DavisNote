#include "patient.h"

#include "list.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int ReadBirthday(FILE *patient_file, struct Date *birthday) {
    if(fscanf(patient_file, "\t%d/%d/%d", &birthday->day, &birthday->month, &birthday->year) != 3) {
       fprintf(stderr, "An error occurred while reading a patient's birthday.\n");
       return 1;
    }
    return 0;
}

int ReadSex(FILE *patient_file, enum Sex *sex) {
    char sex_code;
    if(fscanf(patient_file, "\t%c\n", &sex_code) != 1) {
        fprintf(stderr, "An error occurred while reading a patient's sex.\n");
        return 1;
    }

    switch (sex_code)
    {
    case 'f':
        *sex = Female;
        break;
    case 'M':
        *sex = Male;
        break;
    default:
        fprintf(stderr, "A patient's sex was not one of 'M' or 'F'. It was '%c'\n", sex_code);
        return 1;
    }
    return 0;
}

/*
 * Read patient data from the provided stream.
 */
struct Patient ReadPatient(FILE *patient_file) {
    struct Patient patient;
    memset(&patient, 0, sizeof(struct Patient));
    size_t zero = 0;
    ssize_t length = getline(&patient.name, &zero, patient_file);
    if(length == -1) {
        perror("An error occurred while reading a patient's name.");
        exit(1);
    }
    // Discard the newline read by getline
    patient.name[length - 1] = '\0';

    if(ReadSex(patient_file, &patient.sex) != 0) {
        fprintf(stderr, "An error occurred while reading %s's sex\n", patient.name);
        exit(1);
    }

    if(ReadBirthday(patient_file, &patient.birthday)) {
        fprintf(stderr, "An error occurred while reading %s's birthday\n", patient.name);
        exit(1);
    }

    // Read the list of this patients conditions
    patient.num_conditions = ReadListItems(patient_file, &patient.conditions);
    if(patient.num_conditions == -1) {
        fprintf(stderr, "An error occurred while reading %s's conditions\n", patient.name);
        exit(1);
    }

    // Read the list of this patients medications
    patient.num_medications = ReadListItems(patient_file, &patient.medications);
    if(patient.num_medications == -1) {
        fprintf(stderr, "An error occurred while reading %s's medications\n", patient.name);
        exit(1);
    }

    return patient;
}

struct Patient *ReadAllPatients(const char *patient_data_filename, size_t *num_patients) {
    FILE *patients_file = fopen(patient_data_filename, "r");
    if(!patients_file) {
        perror("Could not open patient data file.");
        exit(1);
    }
    fscanf(patients_file, "%zu\n", num_patients);

    struct Patient *all_patients = malloc(*num_patients * sizeof(struct Patient));

    for(int i = 0; i < *num_patients; i++) {
        all_patients[i] = ReadPatient(patients_file);
    }

    fclose(patients_file);
    return all_patients;
}
