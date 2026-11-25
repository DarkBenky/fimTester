function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function inline(text) {
  let result = escapeHtml(text)
  result = result.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
  result = result.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  result = result.replace(/\*(.+?)\*/g, '<em>$1</em>')
  result = result.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>')
  // Only http(s) links survive. Everything else is rendered as plain text, which
  // keeps javascript: URLs from model output harmless.
  result = result.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (match, label, url) => {
    if (!/^https?:\/\//i.test(url)) return label
    return `<a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a>`
  })
  return result
}

// formatMarkdown turns model output into HTML block by block. Rendering complete
// blocks avoids the empty paragraphs that a chain of global replaces produces, which
// is what made long summaries look like they had holes in them.
export function formatMarkdown(text) {
  if (!text) return ''

  const lines = String(text).replace(/\r\n/g, '\n').split('\n')
  const out = []
  let paragraph = []
  let list = null
  let listTag = ''
  let ordered = false

  const flushParagraph = () => {
    if (!paragraph.length) return
    out.push(`<p>${paragraph.map(inline).join('<br>')}</p>`)
    paragraph = []
  }
  const flushList = () => {
    if (!list) return
    out.push(`<${listTag}>${list.join('')}</${listTag}>`)
    list = null
  }
  const closeBlocks = () => {
    flushParagraph()
    flushList()
  }

  for (const raw of lines) {
    const line = raw.trimEnd()
    const trimmed = line.trim()

    if (!trimmed) {
      closeBlocks()
      continue
    }
    if (/^([-*_])\1{2,}$/.test(trimmed)) {
      closeBlocks()
      out.push('<hr>')
      continue
    }

    const heading = trimmed.match(/^(#{1,4})\s+(.*)$/)
    if (heading) {
      closeBlocks()
      const level = Math.min(heading[1].length + 1, 5)
      out.push(`<h${level}>${inline(heading[2])}</h${level}>`)
      continue
    }

    const bullet = trimmed.match(/^[-*+]\s+(.*)$/)
    if (bullet) {
      flushParagraph()
      if (!list || ordered) {
        flushList()
        list = []
        listTag = 'ul'
        ordered = false
      }
      list.push(`<li>${inline(bullet[1])}</li>`)
      continue
    }

    const numbered = trimmed.match(/^\d+[.)]\s+(.*)$/)
    if (numbered) {
      flushParagraph()
      if (!list || !ordered) {
        flushList()
        list = []
        listTag = 'ol'
        ordered = true
      }
      list.push(`<li>${inline(numbered[1])}</li>`)
      continue
    }

    // A line indented under a list item continues that item.
    if (list && /^\s{2,}/.test(line)) {
      const last = list.length - 1
      list[last] = list[last].replace(/<\/li>$/, ` ${inline(trimmed)}</li>`)
      continue
    }

    flushList()
    paragraph.push(trimmed)
  }
  closeBlocks()
  return out.join('')
}
