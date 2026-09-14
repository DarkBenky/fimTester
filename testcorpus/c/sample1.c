#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int sum_array(const int *values, size_t count) {
    int total = 0;
    for (size_t i = 0; i < count; i++) {
        total += values[i];
    }
    return total;
}

int filter_min(const int *values, size_t count, int min, int *out) {
    size_t n = 0;
    for (size_t i = 0; i < count; i++) {
        if (values[i] > min) {
            out[n++] = values[i];
        }
    }
    return (int)n;
}

int main(int argc, char **argv) {
    const char *name = (argc > 1) ? argv[1] : "world";
    printf("hello %s\n", name);
    int nums[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int out[8];
    int n = filter_min(nums, 8, 4, out);
    printf("sum %d\n", sum_array(out, (size_t)n));
    return 0;
}
