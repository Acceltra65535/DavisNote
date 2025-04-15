/*
 * Program to test the calculator program 
 */
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include "calc.h"
#include <math.h>

#define NUM_TESTS 13

struct pair{
    char* input;
    double output;

};

struct pair input_output_pairs[NUM_TESTS] = {
    {"1 + 1", 2.0},
    {"2 * 3", 6.0},
    {"10 - 5", 5.0},
    {"20 / 4", 5.0},
    {"(2 + 3) * 4", 20.0},
    {"10 + (5 - 3) * 2", 14.0},
    {"100 / (5 + 5)", 10.0},
    {"2 * (3 + 4) - 1", 13.0},
    {"(8 + 2) * (3 - 1)", 20.0},
    {"15 - 3 * (2 + 2)", 3.0},
    {"0.1 + 0.2", 0.3},
    {"3.14 * 3", 9.42},
    {"10 / 3", 3.333}
};

bool is_equal(double a, double b) {
    return fabs(a - b) < 0.01;
}

// Given the index of an input-output pair, run the test on the calculator
// program. Returns true if the tests passes, false otherwise.
bool run_test(int test_num) {
    char* input = input_output_pairs[test_num].input;
    double expected = input_output_pairs[test_num].output;
    double result = calc(input);
    
    if (is_equal(result, expected)) {
        printf("Testing %d passed: %s = %.4f\n", test_num + 1, input, result);
        return true;
    } else {
        printf("Testing %d failed: %s. Expected %.4f, Return: %.4f\n", test_num + 1, input, expected, result);
        return false;
    }
}

int main(int argc, char **argv) {
  int num_passed = 0;
  for(int i = 0; i < NUM_TESTS; i++) {
    if(run_test(i))
      num_passed++;
  }
  printf("%i out of %i tests passed.\n", num_passed, NUM_TESTS);
}