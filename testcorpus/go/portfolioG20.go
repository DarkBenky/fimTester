package prices

// Downsample folds consecutive candles so a chart never receives more points than it
// can draw. Aggregation keeps open of the first bar, close of the last, the extremes
// and the summed volume, so the shape of the series is unchanged.
func Downsample(candles []Candle, maxPoints int) []Candle {
	if maxPoints <= 1 || len(candles) <= maxPoints {
		return candles
	}
	factor := (len(candles) + maxPoints - 1) / maxPoints
	result := make([]Candle, 0, (len(candles)+factor-1)/factor)
	for start := 0; start < len(candles); start += factor {
		end := min(start+factor, len(candles))
		bucket := candles[start]
		bucket.Volume = 0
		for i := start; i < end; i++ {
			bar := candles[i]
			if bar.High > bucket.High {
				bucket.High = bar.High
			}
			if bar.Low < bucket.Low {
				bucket.Low = bar.Low
			}
			bucket.Close = bar.Close
			bucket.Volume += bar.Volume
		}
		result = append(result, bucket)
	}
	return result
}
