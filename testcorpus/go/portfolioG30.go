package market

import (
	"fmt"
	"net/url"
	"strings"

	"golang.org/x/net/html"
)

const articleTextLimit = 8000

// PageText reads a link the user pasted. Only http and https are accepted; the
// shared transport already refuses private and loopback destinations.
func (c *Client) PageText(target string) (string, string, error) {
	target = strings.TrimSpace(target)
	parsed, err := url.Parse(target)
	if err != nil || parsed.Host == "" {
		return "", "", fmt.Errorf("market: invalid link")
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", "", fmt.Errorf("market: only http and https links can be read")
	}
	text := c.ArticleText(target)
	if text == "" {
		return "", target, fmt.Errorf("market: the page could not be read")
	}
	title := parsed.Path
	if title == "" || title == "/" {
		title = parsed.Host
	}
	return text, parsed.Host + title, nil
}

// ArticleText downloads a news page and keeps its readable paragraphs. Full article
// text feeds summaries and search results, so boilerplate is dropped early.
func (c *Client) ArticleText(url string) string {
	if url == "" {
		return ""
	}
	page, err := c.getTextHTML(url)
	if err != nil {
		return ""
	}
	doc, err := html.Parse(strings.NewReader(page))
	if err != nil {
		return ""
	}

	skip := map[string]bool{"script": true, "style": true, "nav": true, "footer": true, "header": true, "aside": true, "form": true}
	var parts []string
	size := 0
	var visit func(node *html.Node)
	visit = func(node *html.Node) {
		if size >= articleTextLimit {
			return
		}
		if node.Type == html.ElementNode && skip[node.Data] {
			return
		}
		if node.Type == html.ElementNode && node.Data == "p" {
			text := strings.TrimSpace(textOf(node))
			if len(text) > 40 {
				parts = append(parts, text)
				size += len(text)
			}
			return
		}
		for child := node.FirstChild; child != nil; child = child.NextSibling {
			visit(child)
		}
	}
	visit(doc)

	text := strings.Join(parts, " ")
	if len(text) > articleTextLimit {
		text = text[:articleTextLimit]
	}
	return text
}

func (c *Client) getTextHTML(target string) (string, error) {
	req, err := newRequest(target)
	if err != nil {
		return "", err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return "", errStatus(resp.StatusCode)
	}
	buffer := new(strings.Builder)
	buffer.Grow(64 << 10)
	chunk := make([]byte, 32<<10)
	total := 0
	for total < 2<<20 {
		n, readErr := resp.Body.Read(chunk)
		if n > 0 {
			buffer.Write(chunk[:n])
			total += n
		}
		if readErr != nil {
			break
		}
	}
	return buffer.String(), nil
}

type statusError int

func (s statusError) Error() string { return "http status" }

func errStatus(code int) error { return statusError(code) }
