package main

import (
	"errors"
	"fmt"
)

type stack struct {
	items []int
}

func (s *stack) push(value int) {
	s.items = append(s.items, value)
}

func (s *stack) pop() (int, error) {
	if len(s.items) == 0 {
		return 0, errors.New("pop from empty stack")
	}
	last := len(s.items) - 1
	value := s.items[last]
	s.items = s.items[:last]
	return value, nil
}

func (s *stack) peek() (int, error) {
	if len(s.items) == 0 {
		return 0, errors.New("peek on empty stack")
	}
	return s.items[len(s.items)-1], nil
}

func main() {
	s := &stack{}
	for i := 1; i <= 5; i++ {
		s.push(i * i)
	}
	for len(s.items) > 0 {
		value, err := s.pop()
		if err != nil {
			fmt.Println("error:", err)
			break
		}
		fmt.Println("popped", value)
	}
	if _, err := s.peek(); err != nil {
		fmt.Println("empty:", err)
	}
}
