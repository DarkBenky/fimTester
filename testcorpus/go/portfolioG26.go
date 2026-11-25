package prices

import (
	"path/filepath"
	"testing"
	"time"
)

func countRows(t *testing.T, store *Store) int64 {
	t.Helper()
	var rows int64
	if err := store.db.QueryRow(`SELECT COUNT(*) FROM prices`).Scan(&rows); err != nil {
		t.Fatalf("count: %v", err)
	}
	return rows
}

// Compaction runs on every boot, so a second pass must not keep shrinking the table.
func TestCompactIsIdempotent(t *testing.T) {
	store, err := Init(filepath.Join(t.TempDir(), "prices.duckdb"), nil, 500)
	if err != nil {
		t.Fatalf("init: %v", err)
	}
	defer store.Close()

	now := time.Now().UTC().Truncate(24 * time.Hour)
	var bars []Bar
	for day := 0; day < 400; day++ {
		base := now.AddDate(0, 0, -day).Unix()
		for step := 0; step < 24; step++ {
			ts := base + int64(step)*3600
			bars = append(bars, Bar{Ticker: "TEST", Date: ts, Open: 10, High: 12, Low: 9, Close: 11, Volume: 100})
		}
	}
	if err := store.Upsert(bars); err != nil {
		t.Fatalf("upsert: %v", err)
	}

	if err := store.Compact(14, 120); err != nil {
		t.Fatalf("first compact: %v", err)
	}
	first := countRows(t, store)
	if err := store.Compact(14, 120); err != nil {
		t.Fatalf("second compact: %v", err)
	}
	second := countRows(t, store)
	if first != second {
		t.Fatalf("compaction is not idempotent: %d rows then %d rows", first, second)
	}

	var open, high, low, closePrice float64
	var volume int64
	row := store.db.QueryRow(`SELECT open, high, low, close, volume FROM prices
		ORDER BY date ASC LIMIT 1`)
	if err := row.Scan(&open, &high, &low, &closePrice, &volume); err != nil {
		t.Fatalf("scan first bar: %v", err)
	}
	if open != 10 || high != 12 || low != 9 || closePrice != 11 || volume != 2400 {
		t.Fatalf("daily aggregate lost data: %v %v %v %v %v", open, high, low, closePrice, volume)
	}
}
