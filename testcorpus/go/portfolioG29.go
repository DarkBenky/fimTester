package market

import (
	"strings"
	"unicode"
)

// Finance sentiment is scored with a small weighted lexicon instead of a transformer
// model: it costs nothing, runs in microseconds and is accurate enough to rank news
// for the dashboard.
var positiveWords = map[string]float64{
	"beat": 1, "beats": 1, "surge": 1.2, "surges": 1.2, "soar": 1.2, "soars": 1.2,
	"rally": 1, "rallies": 1, "gain": 0.8, "gains": 0.8, "rise": 0.6, "rises": 0.6,
	"growth": 0.8, "profit": 0.8, "record": 0.6, "upgrade": 1, "upgraded": 1,
	"outperform": 1, "buyback": 0.8, "dividend": 0.4, "expansion": 0.6, "strong": 0.6,
	"approval": 0.8, "approved": 0.8, "partnership": 0.6, "launch": 0.4, "wins": 0.8,
	"bullish": 1, "optimistic": 0.8, "recovery": 0.6, "boost": 0.8, "raises": 0.6,
	"exceed": 1, "exceeds": 1, "healthy": 0.5, "upside": 0.8, "breakthrough": 1,
}

var negativeWords = map[string]float64{
	"miss": 1, "misses": 1, "plunge": 1.2, "plunges": 1.2, "slump": 1, "slumps": 1,
	"fall": 0.6, "falls": 0.6, "drop": 0.6, "drops": 0.6, "decline": 0.8,
	"loss": 0.8, "losses": 0.8, "downgrade": 1, "downgraded": 1, "underperform": 1,
	"lawsuit": 0.8, "investigation": 0.8, "probe": 0.6, "recall": 0.8, "layoff": 0.8,
	"layoffs": 0.8, "warning": 0.8, "weak": 0.6, "bankruptcy": 1.4, "fraud": 1.4,
	"bearish": 1, "pessimistic": 0.8, "cut": 0.6, "cuts": 0.6, "risk": 0.4,
	"sanction": 0.8, "sanctions": 0.8, "downturn": 1, "crash": 1.2,
}

var negations = map[string]bool{"not": true, "no": true, "never": true, "without": true, "hardly": true}

// ScoreSentiment returns a value in [-1, 1] with title and summary weighted heavier
// than the article body.
func ScoreSentiment(title string, summary string, text string) float64 {
	titleScore := scoreText(title)
	summaryScore := scoreText(summary)
	textScore := scoreText(text)
	return clamp((0.4*titleScore + 0.35*summaryScore + 0.25*textScore), -1, 1)
}

func scoreText(value string) float64 {
	if value == "" {
		return 0
	}
	words := strings.FieldsFunc(strings.ToLower(value), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '\''
	})
	hits, total := 0.0, 0.0
	negated := false
	for _, word := range words {
		if negations[word] {
			negated = true
			continue
		}
		weight, positive := positiveWords[word]
		if !positive {
			if negative, ok := negativeWords[word]; ok {
				weight, positive = negative, false
			} else {
				negated = false
				continue
			}
		}
		if negated {
			positive = !positive
			negated = false
		}
		hits++
		if positive {
			total += weight
		} else {
			total -= weight
		}
	}
	if hits == 0 {
		return 0
	}
	return clamp(total/hits, -1, 1)
}

func clamp(value, low, high float64) float64 {
	if value < low {
		return low
	}
	if value > high {
		return high
	}
	return value
}
