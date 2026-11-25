package config

import (
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/joho/godotenv"
)

type Config struct {
	DevMode       bool
	ServerHost    string
	GoPort        string
	Salt          string
	JWTSecret     string
	JWTExpiry     time.Duration
	DataDir       string
	SQLitePath    string
	PriceDBPath   string
	OpenRouterKey string

	ChatModel         string
	SessionModel      string
	CheapModel        string
	EmbeddingModel    string
	LLMDailyBudgetUSD float64
	LLMDataDir        string

	PriceRetention5mDays int
	PriceRetention1hDays int
	PriceLookback5mDays  int
	ChartMaxPoints       int

	EnableEmbeddings bool
	EmbedArticles    bool
	EmbedDimensions  int
	HTTPAddr         string
	ExtraHTTPPorts   []string
}

func Load() Config {
	if err := godotenv.Load(); err != nil {
		log.Printf("no .env file loaded: %v", err)
	}

	cfg := Config{
		DevMode:              os.Getenv("DEV_MODE") == "true",
		ServerHost:           envOr("SERVER_HOST", "localhost"),
		GoPort:               envOr("BACKEND_GO_PORT", "8085"),
		Salt:                 os.Getenv("SALT"),
		JWTSecret:            os.Getenv("JWT_SECRET"),
		JWTExpiry:            7 * 24 * time.Hour,
		DataDir:              envOr("DATA_DIR", "."),
		SQLitePath:           envOr("SQLITE_PATH", "./portfolio.db"),
		PriceDBPath:          envOr("PRICE_DB_PATH", "./prices.duckdb"),
		OpenRouterKey:        os.Getenv("OPENROUTER_API_KEY"),
		ChatModel:            envOr("OPENROUTER_MODEL", "qwen/qwen3.7-flash"),
		SessionModel:         envOr("OPENROUTER_CHAT_MODEL", "tencent/hy4-flash"),
		CheapModel:           envOr("OPENROUTER_CHEAP_MODEL", "inclusionai/ling-3.0-flash"),
		EmbeddingModel:       envOr("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
		LLMDailyBudgetUSD:    envFloat("LLM_DAILY_BUDGET_USD", 0.5),
		PriceRetention5mDays: envInt("PRICE_RETENTION_5M_DAYS", 14),
		PriceRetention1hDays: envInt("PRICE_RETENTION_1H_DAYS", 120),
		PriceLookback5mDays:  envInt("PRICE_LOOKBACK_5M_DAYS", 14),
		ChartMaxPoints:       envInt("CHART_MAX_POINTS", 500),
		EnableEmbeddings:     os.Getenv("RAG_USE_EMBEDDINGS") == "1",
		EmbedArticles:        os.Getenv("RAG_EMBED_ARTICLES") == "1",
		EmbedDimensions:      envInt("RAG_EMBED_DIMENSIONS", 512),
	}

	if hours := os.Getenv("JWT_EXPIRY_HOURS"); hours != "" {
		value, err := strconv.Atoi(hours)
		if err != nil || value <= 0 {
			log.Fatal("JWT_EXPIRY_HOURS must be a positive integer")
		}
		cfg.JWTExpiry = time.Duration(value) * time.Hour
	}
	if cfg.Salt == "" || cfg.JWTSecret == "" {
		log.Fatal("Required environment variables (SALT, JWT_SECRET) not set")
	}

	cfg.HTTPAddr = ":" + cfg.GoPort
	// The UI used to be served by a separate process on 5173, so the same router is
	// also exposed there to keep the existing URL working.
	for _, port := range strings.Split(os.Getenv("EXTRA_HTTP_PORTS"), ",") {
		if trimmed := strings.TrimSpace(port); trimmed != "" {
			cfg.ExtraHTTPPorts = append(cfg.ExtraHTTPPorts, trimmed)
		}
	}
	return cfg
}

func (c Config) AllowedOrigins() []string {
	if c.DevMode {
		return []string{"http://localhost:5173", "http://127.0.0.1:5173"}
	}
	return []string{"http://" + c.ServerHost + ":5173", "http://" + c.ServerHost + ":" + c.GoPort}
}

func envOr(name string, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func envInt(name string, fallback int) int {
	value, err := strconv.Atoi(os.Getenv(name))
	if err != nil || value <= 0 {
		return fallback
	}
	return value
}

func envFloat(name string, fallback float64) float64 {
	value, err := strconv.ParseFloat(os.Getenv(name), 64)
	if err != nil || value < 0 {
		return fallback
	}
	return value
}
