package main

import (
	"fmt"
	"sort"
	"strings"
)

type wordCount struct {
	word  string
	count int
}

func countWords(text string) map[string]int {
	counts := map[string]int{}
	for _, word := range strings.Fields(strings.ToLower(text)) {
		word = strings.Trim(word, ".,!?;:")
		if word != "" {
			counts[word]++
		}
	}
	return counts
}

func topWords(counts map[string]int, limit int) []wordCount {
	ranked := make([]wordCount, 0, len(counts))
	for word, count := range counts {
		ranked = append(ranked, wordCount{word: word, count: count})
	}
	sort.Slice(ranked, func(i, j int) bool {
		if ranked[i].count == ranked[j].count {
			return ranked[i].word < ranked[j].word
		}
		return ranked[i].count > ranked[j].count
	})
	if limit < len(ranked) {
		return ranked[:limit]
	}
	return ranked
}

func main() {
	text := "the quick brown fox jumps over the lazy dog the fox"
	for _, entry := range topWords(countWords(text), 3) {
		fmt.Printf("%s: %d\n", entry.word, entry.count)
	}
}
