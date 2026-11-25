package main

import (
	"database/sql"
	"fmt"
	"os"
	"time"

	"portfolio/internal/prices"
)

func count(db *sql.DB) int64 {
	var rows int64
	if err := db.QueryRow(`SELECT COUNT(*) FROM prices`).Scan(&rows); err != nil {
		panic(err)
	}
	return rows
}

func main() {
	path := os.Args[1]
	store, err := prices.Init(path, nil, 500)
	if err != nil {
		panic(err)
	}
	for pass := 1; pass <= 3; pass++ {
		started := time.Now()
		if err := store.Compact(14, 120); err != nil {
			panic(err)
		}
		fmt.Printf("pass %d took %s\n", pass, time.Since(started).Round(time.Second))
		if err := store.Checkpoint(); err != nil {
			panic(err)
		}
	}
	if err := store.Close(); err != nil {
		panic(err)
	}
	db, err := sql.Open("duckdb", path)
	if err != nil {
		panic(err)
	}
	defer db.Close()
	db.SetMaxOpenConns(1)
	fmt.Println("final rows", count(db))
	var oldest string
	db.QueryRow(`SELECT MIN(to_timestamp(date)) FROM prices`).Scan(&oldest)
	fmt.Println("oldest", oldest)
}
