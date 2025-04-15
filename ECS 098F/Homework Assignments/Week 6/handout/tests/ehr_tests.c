/*
 * End-to-end tests for the EHR program using the CUnit framework.
 */
#include <CUnit/Basic.h>
#include <stdio.h>
#include <stdlib.h>

// TestPatient 1 through 3 are testing the "patients.1.txt" file

void TestPatient1()
{
    FILE *output_stream = popen("echo \"John Doe\" | ./ehr sample_data/patients.1.txt", "r");
    CU_ASSERT(output_stream != NULL);

    char *expected_output[7] = {
        "Enter a patient's name: Patient: John Doe\n",
        "\tSex: M\n",
        "\tDate of Birth: 7/16/1984\n",
        "Diagnosed conditions:\n",
        "\tFacial Agoricosis, Agpentive Colon\n",
        "Perscribed medications:\n",
        "\tPancrebalamin\n"};

    char buffer[256];
    size_t line_index = 0;
    while (fgets(buffer, sizeof(buffer), output_stream) != NULL)
    {
        CU_ASSERT(strcmp(buffer, expected_output[line_index]) == 0);
        line_index++;
    }

    // Assert that the program output had as many lines as the expected output
    CU_ASSERT(line_index == sizeof(expected_output) / sizeof(char *));
    // Assert that the program exited with code 0
    CU_ASSERT(pclose(output_stream) == 0)
}

void TestPatient2()
{
    FILE *output_stream = popen("echo \"Jane Foe\" | ./ehr sample_data/patients.1.txt", "r");
    CU_ASSERT(output_stream != NULL);

    char *expected_output[7] = {
        "Enter a patient's name: Patient: Jane Foe\n",
        "\tSex: F\n",
        "\tDate of Birth: 1/31/2000\n",
        "Diagnosed conditions:\n",
        "\tNone\n",
        "Perscribed medications:\n",
        "\tNone\n"};

    char buffer[256];
    size_t line_index = 0;
    while (fgets(buffer, sizeof(buffer), output_stream) != NULL)
    {
        CU_ASSERT(strcmp(buffer, expected_output[line_index]) == 0);
        line_index++;
    }

    // Assert that the program output had as many lines as the expected output
    CU_ASSERT(line_index == sizeof(expected_output) / sizeof(char *));
    // Assert that the program exited with code 0
    CU_ASSERT(pclose(output_stream) == 0)
}

void TestPatient3()
{
    FILE *output_stream = popen("echo \"Phenste Tot\" | ./ehr sample_data/patients.1.txt", "r");
    CU_ASSERT(output_stream != NULL);

    char *expected_output[7] = {
        "Enter a patient's name: Patient: Phenste Tot\n",
        "\tSex: M\n",
        "\tDate of Birth: 2/28/1999\n",
        "Diagnosed conditions:\n",
        "\tBacterial Fradular Syndrome\n",
        "Perscribed medications:\n",
        "\tAmacelex, Benzocerol Atracumulin, Oratosine\n"};

    char buffer[256];
    size_t line_index = 0;
    while (fgets(buffer, sizeof(buffer), output_stream) != NULL)
    {
        CU_ASSERT(strcmp(buffer, expected_output[line_index]) == 0);
        line_index++;
    }

    // Assert that the program output had as many lines as the expected output
    CU_ASSERT(line_index == sizeof(expected_output) / sizeof(char *));
    // Assert that the program exited with code 0
    CU_ASSERT(pclose(output_stream) == 0)
}

// TestPatient 4 through 6 tests the "patients.2.txt" file

void TestPatient4()
{
    FILE *output_stream = popen("echo \"Adem Wong\" | ./ehr sample_data/patients.2.txt", "r");
    CU_ASSERT(output_stream != NULL);

    char *expected_output[7] = {
        "Enter a patient's name: Patient: Adem Wong\n",
        "\tSex: M\n",
        "\tDate of Birth: 9/1/1801\n",
        "Diagnosed conditions:\n",
        "\tPestilent Mutation, Pig Scarring, Violet Pneumonia, Winter Scurvy, Running Shock\n",
        "Perscribed medications:\n",
        "\tBaracotrol, Phoslin, Sulfaruvax Norxone\n"};

    char buffer[256];
    size_t line_index = 0;
    while (fgets(buffer, sizeof(buffer), output_stream) != NULL)
    {
        CU_ASSERT(strcmp(buffer, expected_output[line_index]) == 0);
        line_index++;
    }

    // Assert that the program output had as many lines as the expected output
    CU_ASSERT(line_index == sizeof(expected_output) / sizeof(char *));
    // Assert that the program exited with code 0
    CU_ASSERT(pclose(output_stream) == 0)
}

void TestPatient5()
{
    FILE *output_stream = popen("echo \"Faizaan Blaese\" | ./ehr sample_data/patients.2.txt", "r");
    CU_ASSERT(output_stream != NULL);

    char *expected_output[7] = {
        "Enter a patient's name: Patient: Faizaan Blaese\n",
        "\tSex: M\n",
        "\tDate of Birth: 11/14/1988\n",
        "Diagnosed conditions:\n",
        "\tWater Nausea, Milk Throat, Decaying Gonorrhea, Man Stiffness\n",
        "Perscribed medications:\n",
        "\tKanupizole, Tamobucil, Acycterol, Vivoretin, Baracotrol, Phoslin, Sulfaruvax Norxone, Abareset Pazolovir, Olmedosyn Ocupirin, Alimvac Adrenanovate\n"};

    char buffer[256];
    size_t line_index = 0;
    while (fgets(buffer, sizeof(buffer), output_stream) != NULL)
    {
        CU_ASSERT(strcmp(buffer, expected_output[line_index]) == 0);
        line_index++;
    }

    // Assert that the program output had as many lines as the expected output
    CU_ASSERT(line_index == sizeof(expected_output) / sizeof(char *));
    // Assert that the program exited with code 0
    CU_ASSERT(pclose(output_stream) == 0)
}

void TestPatient6()
{
    FILE *output_stream = popen("echo \"Yasmeen Ortiz-Salinas\" | ./ehr sample_data/patients.2.txt", "r");
    CU_ASSERT(output_stream != NULL);

    char *expected_output[7] = {
        "Enter a patient's name: Patient: Yasmeen Ortiz-Salinas\n",
        "\tSex: F\n",
        "\tDate of Birth: 2/1/2002\n",
        "Diagnosed conditions:\n",
        "\tEthereal Cramps\n",
        "Perscribed medications:\n",
        "\tNone\n"};

    char buffer[256];
    size_t line_index = 0;
    while (fgets(buffer, sizeof(buffer), output_stream) != NULL)
    {
        CU_ASSERT(strcmp(buffer, expected_output[line_index]) == 0);
        line_index++;
    }

    // Assert that the program output had as many lines as the expected output
    CU_ASSERT(line_index == sizeof(expected_output) / sizeof(char *));
    // Assert that the program exited with code 0
    CU_ASSERT(pclose(output_stream) == 0)
}

void TestPatientNotFound()
{
    FILE *output_stream = popen("echo \"John Lennon\" | ./ehr sample_data/patients.1.txt", "r");
    CU_ASSERT(output_stream != NULL);

    char *expected_output[1] = {
        "Enter a patient's name: No patient with the name 'John Lennon' could be found.\n"};

    char buffer[256];
    size_t line_index = 0;
    while (fgets(buffer, sizeof(buffer), output_stream) != NULL)
    {
        CU_ASSERT(strcmp(buffer, expected_output[line_index]) == 0);
        line_index++;
    }

    // Assert that the program output had as many lines as the expected output
    CU_ASSERT(line_index == sizeof(expected_output) / sizeof(char *));
    // Assert that the program did not exit with code 0
    CU_ASSERT(pclose(output_stream) != 0)
}

int main(void)
{
    // Initialize CUnit
    CU_initialize_registry();
    CU_pSuite pSuite = CU_add_suite("EhrTests", NULL, NULL);

    // Register unit tests
    CU_add_test(pSuite, "TestPatient1", TestPatient1);
    CU_add_test(pSuite, "TestPatient2", TestPatient2);
    CU_add_test(pSuite, "TestPatient3", TestPatient3);
    CU_add_test(pSuite, "TestPatient4", TestPatient4);
    CU_add_test(pSuite, "TestPatient5", TestPatient5);
    CU_add_test(pSuite, "TestPatient6", TestPatient6);
    CU_add_test(pSuite, "TestPatientNotFound", TestPatientNotFound);

    // Run all tests
    CU_basic_run_tests();
    // Print number of tests passed
    fprintf(stderr, "%d", CU_get_number_of_tests_run() - CU_get_number_of_tests_failed());

    // Clean up & exit
    CU_cleanup_registry();
    return 0;
}
