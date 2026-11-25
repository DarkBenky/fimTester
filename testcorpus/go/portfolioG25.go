//go:build unix

package api

import (
	"os"
	"path/filepath"
	"syscall"
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
	usage := diskUsage{Path: path}
	var stat syscall.Statfs_t
	if err := syscall.Statfs(path, &stat); err != nil {
		usage.Error = err.Error()
		return usage
	}
	blockSize := uint64(stat.Bsize)
	usage.TotalBytes = stat.Blocks * blockSize
	usage.FreeBytes = stat.Bavail * blockSize
	if usage.TotalBytes > usage.FreeBytes {
		usage.UsedBytes = usage.TotalBytes - usage.FreeBytes
	}
	if usage.TotalBytes > 0 {
		usage.UsedPct = float64(usage.UsedBytes) / float64(usage.TotalBytes) * 100
	}
	return usage
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
