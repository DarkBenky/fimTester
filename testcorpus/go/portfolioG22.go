//go:build !unix

package api

import (
	"os"
	"path/filepath"
)

type diskUsage struct {
	Path       string  `json:"path"`
	TotalBytes uint64  `json:"total_bytes"`
	FreeBytes  uint64  `json:"free_bytes"`
	UsedBytes  uint64  `json:"used_bytes"`
	UsedPct    float64 `json:"used_percent"`
	Error      string  `json:"error,omitempty"`
}

func readDisk(path string) diskUsage {
	return diskUsage{Path: path, Error: "disk usage is only available on unix"}
}

type fileSize struct {
	Name  string `json:"name"`
	Bytes int64  `json:"bytes"`
}

func directoryFiles(paths []string) []fileSize {
	files := make([]fileSize, 0, len(paths))
	for _, path := range paths {
		info, err := os.Stat(path)
		if err != nil {
			files = append(files, fileSize{Name: filepath.Base(path)})
			continue
		}
		files = append(files, fileSize{Name: filepath.Base(path), Bytes: info.Size()})
	}
	return files
}
