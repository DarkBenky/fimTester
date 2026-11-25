package main

import (
	"fmt"
	"log"
	"time"

	zmq "github.com/pebbe/zmq4"
)

const (
	pullAddr = "tcp://*:5555"
	pubAddr  = "tcp://*:5556"
)

func main() {
	pull, err := zmq.NewSocket(zmq.PULL)
	if err != nil {
		log.Fatalf("Failed to create PULL socket: %v", err)
	}
	defer pull.Close()
	// Keep only the latest frame in the receive buffer; drop older ones.
	if err := pull.SetConflate(true); err != nil {
		log.Fatalf("Failed to set CONFLATE on PULL socket: %v", err)
	}
	if err := pull.SetRcvhwm(1); err != nil {
		log.Fatalf("Failed to set RCVHWM on PULL socket: %v", err)
	}
	if err := pull.Bind(pullAddr); err != nil {
		log.Fatalf("Failed to bind PULL socket: %v", err)
	}

	pub, err := zmq.NewSocket(zmq.PUB)
	if err != nil {
		log.Fatalf("Failed to create PUB socket: %v", err)
	}
	defer pub.Close()
	// Only keep the latest frame queued for each subscriber.
	if err := pub.SetConflate(true); err != nil {
		log.Fatalf("Failed to set CONFLATE on PUB socket: %v", err)
	}
	if err := pub.SetSndhwm(1); err != nil {
		log.Fatalf("Failed to set SNDHWM on PUB socket: %v", err)
	}
	if err := pub.Bind(pubAddr); err != nil {
		log.Fatalf("Failed to bind PUB socket: %v", err)
	}

	fmt.Printf("flycam server running\n")
	fmt.Printf("  video PULL %s  PUB %s\n", pullAddr, pubAddr)

	var totalBytes int64
	var frameCount int64
	lastLog := time.Now()

	for {
		data, err := pull.RecvBytes(0)
		if err != nil {
			log.Printf("Recv error: %v", err)
			continue
		}

		if _, err := pub.SendBytes(data, 0); err != nil {
			log.Printf("Pub send error: %v", err)
			continue
		}

		totalBytes += int64(len(data))
		frameCount++

		if elapsed := time.Since(lastLog); elapsed >= time.Second {
			kbps := float64(totalBytes) / elapsed.Seconds() / 1024
			fps := float64(frameCount) / elapsed.Seconds()
			log.Printf("[go]  %.1f KB/s  %.1f fps", kbps, fps)
			totalBytes = 0
			frameCount = 0
			lastLog = time.Now()
		}
	}
}
