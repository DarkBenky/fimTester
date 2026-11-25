package api

import (
	"encoding/json"
	"strconv"
	"time"

	"github.com/labstack/echo/v4"

	"portfolio/internal/prices"
)

type pricesCandle = prices.Candle

func pricesIntervalSupported(interval string) bool {
	return prices.Supported(interval)
}

func analyticsLookback(interval string) int64 {
	return prices.IntervalLookback(interval, 14)
}

func startOfToday() int64 {
	now := time.Now().UTC()
	return time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.UTC).Unix()
}

// intervalBucket snaps a timestamp to the start of the chart bucket so daily sentiment
// values land on the same x axis slot as the candle they describe.
type intervalBucket struct {
	seconds int64
	weekly  bool
	monthly bool
}

func (b intervalBucket) start(timestamp int64) int64 {
	switch {
	case b.weekly:
		return ((timestamp-345600)/604800)*604800 + 345600
	case b.monthly:
		t := time.Unix(timestamp, 0).UTC()
		return time.Date(t.Year(), t.Month(), 1, 0, 0, 0, 0, time.UTC).Unix()
	default:
		if b.seconds <= 0 {
			return timestamp
		}
		return timestamp / b.seconds * b.seconds
	}
}

func intervalBucketFor(interval string) intervalBucket {
	switch interval {
	case "5m":
		return intervalBucket{seconds: 300}
	case "15m":
		return intervalBucket{seconds: 900}
	case "1h":
		return intervalBucket{seconds: 3600}
	case "4h":
		return intervalBucket{seconds: 14400}
	case "1w":
		return intervalBucket{weekly: true}
	case "1M":
		return intervalBucket{monthly: true}
	default:
		return intervalBucket{seconds: 86400}
	}
}

func queryOffset(c echo.Context) int {
	value, err := strconv.Atoi(c.QueryParam("offset"))
	if err != nil || value < 0 {
		return 0
	}
	return value
}

func defaultQuery(c interface{ QueryParam(string) string }, name string, fallback string) string {
	if value := c.QueryParam(name); value != "" {
		return value
	}
	return fallback
}

func jsonUnmarshal(data []byte, out any) error {
	return json.Unmarshal(data, out)
}
