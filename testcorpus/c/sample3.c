#include <stdio.h>
#include <stdlib.h>

static void swap(int *a, int *b) {
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

void sort_desc(int *values, size_t count) {
    for (size_t i = 0; i + 1 < count; i++) {
        for (size_t j = 0; j + 1 < count - i; j++) {
            if (values[j] < values[j + 1]) {
                swap(&values[j], &values[j + 1]);
            }
        }
    }
}

int find_index(const int *values, size_t count, int target) {
    for (size_t i = 0; i < count; i++) {
        if (values[i] == target) {
            return (int)i;
        }
    }
    return -1;
}

int main(void) {
    int data[] = {42, 7, 19, 3, 88, 23, 7};
    size_t n = sizeof(data) / sizeof(data[0]);
    sort_desc(data, n);
    for (size_t i = 0; i < n; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");
    printf("index of 19: %d\n", find_index(data, n, 19));
    return 0;
}
