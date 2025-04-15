#ifndef CALC_H
#define CALC_H

/*
 * calc - Evaluate mathematical expression
 * @expr: String containing mathematical expression
 *
 * This function receives a string describing a mathematical expression, and
 * returns the results of its evaluation.
 *
 * The expression can contain `+`, `-`, `*`, `/`, `(`, `)`, *non-negative*
 * integers, and empty spaces. An example could be `(0 - 3) / 2`.
 */
int calc(char *expr);

#endif /* CALC_H */
