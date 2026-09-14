import random

from tree_sitter import Language, Parser

from files import read_source

GRAMMARS = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "go": "tree_sitter_go",
    "c": "tree_sitter_c",
}

BLOCK_TYPES = {
    "python": {
        "function_definition", "class_definition", "if_statement", "for_statement",
        "while_statement", "with_statement", "try_statement", "match_statement",
    },
    "javascript": {
        "function_declaration", "class_declaration", "if_statement", "for_statement",
        "for_in_statement", "while_statement", "do_statement", "switch_statement",
        "try_statement", "function_expression", "arrow_function",
    },
    "go": {
        "function_declaration", "method_declaration", "if_statement", "for_statement",
        "switch_statement", "type_declaration", "const_declaration", "var_declaration",
    },
    "c": {
        "function_definition", "if_statement", "for_statement", "while_statement",
        "switch_statement", "do_statement",
    },
}

TOP_LEVEL = {"program", "source_file", "module", "translation_unit"}

COMMENT_MARKERS = {
    "python": ("#",),
    "javascript": ("//", "/*", "*"),
    "go": ("//", "/*", "*"),
    "c": ("//", "/*", "*"),
}

MAX_BLOCK_LINES = 40
EDGE_LINES = 3


class Hole:
    def __init__(self, file, language, start_line, end_line, cut, removed_text, prefix, suffix, indent):
        self.file = file
        self.language = language
        self.start_line = start_line
        self.end_line = end_line
        self.cut = cut
        self.removed_text = removed_text
        self.prefix = prefix
        self.suffix = suffix
        self.indent = indent

    def to_sample(self):
        return {
            "file": self.file,
            "language": self.language,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "cut": self.cut,
            "removed_text": self.removed_text,
        }


def parse_file(language, text):
    module = __import__(GRAMMARS[language])
    parser = Parser(Language(module.language()))
    return parser.parse(text.encode("utf-8"))


def block_candidates(tree, language, n_lines):
    types = BLOCK_TYPES[language]
    out = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in types:
            if node.start_point[0] >= EDGE_LINES and node.end_point[0] <= n_lines - 1 - EDGE_LINES:
                if node.end_point[0] - node.start_point[0] + 1 <= MAX_BLOCK_LINES:
                    if not is_edge_top_level(node):
                        out.append(node)
        for child in node.children:
            stack.append(child)
    return out


def is_edge_top_level(node):
    parent = node.parent
    if parent is None or parent.type not in TOP_LEVEL:
        return False
    siblings = [c for c in parent.children if c.type != "comment"]
    return node is siblings[0] or node is siblings[-1]


def make_hole(rng, language, text, min_file_lines, span_min, span_max, cut):
    lines = text.split("\n")
    n = len(lines)
    if n < min_file_lines:
        return None
    if cut == "block":
        return make_block_hole(rng, language, text, lines, n)
    if cut == "lines":
        return make_line_hole(rng, lines, n, span_min, span_max)
    if rng.random() < 0.5:
        hole = make_block_hole(rng, language, text, lines, n)
        if hole is not None:
            return hole
    return make_line_hole(rng, lines, n, span_min, span_max)


def make_line_hole(rng, lines, n, span_min, span_max):
    lo, hi = EDGE_LINES, n - 1 - EDGE_LINES
    if hi - lo < 1:
        return None
    length = rng.randint(span_min, span_max)
    length = max(1, min(length, hi - lo))
    start = rng.randint(lo, hi - length)
    end = start + length
    removed = "\n".join(lines[start:end])
    if not is_meaningful(removed, "python"):
        return None
    prefix = "\n".join(lines[:start]) + "\n"
    suffix = "\n".join(lines[end:])
    return Hole(
        file=None, language=None,
        start_line=start + 1, end_line=end,
        cut="lines", removed_text=removed,
        prefix=prefix, suffix=suffix, indent=indent_of(removed),
    )


def make_block_hole(rng, language, text, lines, n):
    tree = parse_file(language, text)
    candidates = block_candidates(tree, language, n)
    if not candidates:
        return None
    for _ in range(20):
        node = rng.choice(candidates)
        removed = text[node.start_byte:node.end_byte]
        if not is_meaningful(removed, language):
            continue
        prefix = text[:node.start_byte]
        suffix = text[node.end_byte:]
        return Hole(
            file=None, language=None,
            start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1,
            cut="block", removed_text=removed,
            prefix=prefix, suffix=suffix, indent=indent_of(removed),
        )
    return None


def is_meaningful(text, language):
    stripped = text.strip()
    if not stripped:
        return False
    markers = COMMENT_MARKERS.get(language, ("#", "//"))
    for line in stripped.split("\n"):
        line = line.strip()
        if not line:
            continue
        if not any(line.startswith(m) for m in markers):
            return True
    return False


def indent_of(text):
    for line in text.split("\n"):
        if line.strip():
            return len(line) - len(line.lstrip(" "))
    return 0


def trim_context(prefix, suffix, budget_tokens):
    if budget_tokens is None:
        return prefix, suffix, False
    budget_chars = budget_tokens * 4
    if len(prefix) + len(suffix) <= budget_chars:
        return prefix, suffix, False
    keep_prefix = min(len(prefix), budget_chars // 2)
    keep_suffix = min(len(suffix), budget_chars - keep_prefix)
    if keep_prefix < len(prefix):
        cut = prefix.rfind("\n", 0, len(prefix) - keep_prefix)
        prefix = prefix[cut + 1:] if cut != -1 else prefix[-keep_prefix:]
    if keep_suffix < len(suffix):
        cut = suffix.find("\n", len(suffix) - keep_suffix)
        suffix = suffix[:cut] if cut != -1 else suffix[:keep_suffix]
    return prefix, suffix, True


def hole_from_sample(sample, text):
    lines = text.split("\n")
    start = sample["start_line"] - 1
    end = sample["end_line"]
    if sample["cut"] == "lines":
        prefix = "\n".join(lines[:start]) + "\n"
        suffix = "\n".join(lines[end:])
    else:
        start_byte = sum(len(l) + 1 for l in lines[:start])
        end_byte = sum(len(l) + 1 for l in lines[:end]) - 1
        prefix = text[:start_byte]
        suffix = text[end_byte:]
    removed = sample["removed_text"]
    return Hole(
        file=sample["file"], language=sample["language"],
        start_line=sample["start_line"], end_line=sample["end_line"],
        cut=sample["cut"], removed_text=removed,
        prefix=prefix, suffix=suffix, indent=indent_of(removed),
    )
