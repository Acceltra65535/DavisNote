/*
 * Program to determine if two words are anagrams of eachother.
 * The words must contain only alphanumeric characters.
 */
#include <string.h>
#include <stdbool.h>
#include <stdio.h>

bool AreAnagrams(char* word1, char* word2) {
    word2 = strdup(word2);

    // The original code while(word1 ! = NULL) is incorrect. In C++, strings are null-terminated character arrays, but word1 itself is a pointer, and it never becomes NULL.
    while(*word1 != '\0') { // Modify this line.
        char* c = index(word2, *word1);
        if(c == NULL) {
            return false;
        } else {
            *c = '-';
        }
        word1++;
    }
    return true;
}

int main(int argc, char** argv) {
    if(argc != 3) {
        printf("Usage: %s <word> <word>\n", argv[0]);
        return 1;
    }

    bool are_anagrams = AreAnagrams(argv[1], argv[2]);
    printf("%s and %s are %sanagrams of each other.\n", argv[1], argv[2], are_anagrams ? "" : "not ");

    if(are_anagrams) {
        return 0;
    } else {
        return 1;
    }
}

// 原代码中的 while(word1 != NULL) 是不正确的。在 C/C++ 中，字符串是以 null 结尾的字符数组，但 word1 本身是一个指针，它永远不会变成 NULL。
// 正确的做法是检查字符串的结束符 '\0'。所以我们将循环条件改为 while(*word1 != '\0')，这样就能正确地遍历整个字符串。
// 这个修改确保了函数会检查 word1 的每个字符，直到遇到字符串结束符。