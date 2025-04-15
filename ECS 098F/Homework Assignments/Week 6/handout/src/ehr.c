#include "ehr.h"
#include "list.h"
#include "patient.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>


size_t num_patients;
struct Patient *all_patients;

struct Patient *FindPatientByName(char *name) {
    struct Patient *patient = NULL;
    for(int i = 0; i < num_patients; i++) {
        if(strcmp(all_patients[i].name, name) == 0) {
            patient = &all_patients[i];
            break;
        }
    }
    return patient;
}

int main(int argc, char **argv) {
    if(argc != 2) {
        printf("Usage %s <patient_data.txt>\n", argv[0]);
        return 1;
    }
    all_patients = ReadAllPatients(argv[1], &num_patients);

    printf("Enter a patient's name: ");
    char entered_name[16];
    fgets(entered_name, sizeof(entered_name), stdin);
    // Discard the newline read by fgets
    entered_name[strcspn(entered_name, "\n")] = '\0';

    struct Patient *patient = FindPatientByName(entered_name);
    if(patient == NULL) {
        printf("No patient with the name '%s' could be found.\n", entered_name);
        return 1;
    }

    printf("Patient: %s\n\tSex: %c\n\tDate of Birth: %i/%i/%i\n", patient->name, patient->sex == Male ? 'M' : 'F', patient->birthday.day, patient->birthday.month, patient->birthday.year);
    printf("Diagnosed conditions:\n\t");
    if(patient->num_conditions == 0) {
        printf("None\n");
    } else {
        for(int i = 0; i < patient->num_conditions; i++) {
            printf("%s", patient->conditions[i]);
            if(i != patient->num_conditions - 1) {
                printf(", ");
            } else {
                printf("\n");
            }
        }
    }
    printf("Perscribed medications:\n\t");
    if(patient->num_medications == 0) {
        printf("None\n");
    } else {
        for(int i = 0; i < patient->num_medications; i++) {
            printf("%s", patient->medications[i]);
            if(i != patient->num_medications - 1) {
                printf(", ");
            } else {
                printf("\n");
            }
        }
    }
    return 0;
}
