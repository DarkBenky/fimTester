#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} buffer_t;

static int buffer_init(buffer_t *buf, size_t cap) {
    buf->data = malloc(cap);
    if (buf->data == NULL) {
        return -1;
    }
    buf->len = 0;
    buf->cap = cap;
    return 0;
}

static int buffer_push(buffer_t *buf, const char *text) {
    size_t need = strlen(text) + 1;
    if (buf->len + need > buf->cap) {
        size_t next = buf->cap * 2;
        char *grown = realloc(buf->data, next);
        if (grown == NULL) {
            return -1;
        }
        buf->data = grown;
        buf->cap = next;
    }
    memcpy(buf->data + buf->len, text, need);
    buf->len += need - 1;
    return 0;
}

int count_words(const char *text) {
    int words = 0;
    int in_word = 0;
    for (const char *p = text; *p != '\0'; p++) {
        if (isspace((unsigned char)*p)) {
            in_word = 0;
        } else if (!in_word) {
            in_word = 1;
            words++;
        }
    }
    return words;
}

int main(void) {
    buffer_t buf;
    if (buffer_init(&buf, 64) != 0) {
        return 1;
    }
    buffer_push(&buf, "the quick brown fox");
    buffer_push(&buf, " jumps over the lazy dog");
    printf("%s\n", buf.data);
    printf("words: %d\n", count_words(buf.data));
    free(buf.data);
    return 0;
}
