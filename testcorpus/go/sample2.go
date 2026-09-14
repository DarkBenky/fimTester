package main

import (
	"fmt"
	"os"
	"strings"
)

func sum(values []int) int {
	total := 0
	for _, v := range values {
		total += v
	}
	return total
}

func filter(values []int, min int) []int {
	out := []int{}
	for _, v := range values {
		if v > min {
			out = append(out, v)
		}
	}
	return out
}

func main() {
	args := os.Args[1:]
	name := "world"
	if len(args) > 0 {
		name = args[0]
	}
	fmt.Println("hello", name)
	nums := []int{1, 2, 3, 4, 5, 6, 7, 8}
	big := filter(nums, 4)
	fmt.Println(sum(big))
}
