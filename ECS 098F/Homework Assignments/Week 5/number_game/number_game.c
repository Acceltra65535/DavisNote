#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main(int argc, char** argv) {
    srand(time(0));
    int r = rand();
    for(int i = 0; i < 10; i++) {
        printf("Guess between 0-100:\n");
        int guess;
        if(scanf("%i", &guess) != 1 || guess > 100 || guess < 0) {
            printf("Please enter a valid guess.\n");
            while(getchar() != '\n');
            i--;
            continue;
        }
        if(guess == r) {
            printf("You win!\n");
            return 0;
        } else if(guess > r) {
            printf("Your guess is too big.");
        } else {
            printf("Your guess is too small.");
        }

        if(i == 9) {
            printf("\nYou lose!\n");
            return 1;
        } else {
            printf(" (%i tries remaining)\n", 9 - i);
        }
    }
}