package api

import (
	"net/http"
	"os"
	"path/filepath"

	"github.com/labstack/echo/v4"
)

func (s *Server) getStorage(c echo.Context) error {
	dir := s.Cfg.DataDir
	if dir == "" {
		dir = "."
	}
	dir, err := filepath.Abs(dir)
	if err != nil {
		dir = s.Cfg.DataDir
	}
	host, _ := os.Hostname()
	files := directoryFiles([]string{
		s.Cfg.SQLitePath,
		s.Cfg.SQLitePath + "-wal",
		s.Cfg.SQLitePath + "-shm",
		s.Cfg.PriceDBPath,
	})
	var databaseBytes int64
	for _, file := range files {
		databaseBytes += file.Bytes
	}
	return c.JSON(http.StatusOK, map[string]any{
		"disk":           readDisk(dir),
		"files":          files,
		"database_bytes": databaseBytes,
		"directory":      dir,
		"host":           host,
	})
}
