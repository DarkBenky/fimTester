import difflib
import re

from tree_sitter import Language, Parser

from spans import GRAMMARS

FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*$")
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+\.?\d*|[^\sA-Za-z0-9_]+")

WRAP = {
    "python": None,
    "javascript": ("function __fim__() {\n", "\n}"),
    "go": ("package main\n\nfunc __fim__() {\n", "\n}"),
    "c": ("void __fim__(void) {\n", "\n}"),
}


def normalize(text):
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def strip_fences(text):
    lines = text.split("\n")
    while lines and (not lines[0].strip() or FENCE.match(lines[0].strip())):
        lines.pop(0)
    while lines and (not lines[-1].strip() or FENCE.match(lines[-1].strip())):
        lines.pop()
    return "\n".join(lines)


def reindent(text, indent):
    pad = " " * indent
    lines = text.split("\n")
    out = []
    for line in lines:
        if not line.strip():
            out.append("")
        else:
            out.append(pad + line.lstrip(" "))
    return "\n".join(out)


def postprocess(completion, language, indent):
    text = strip_fences(completion)
    if language == "python":
        text = reindent(text, indent)
    return text


def exact_match(a, b):
    return a == b


def similarity(a, b):
    char = difflib.SequenceMatcher(None, a, b).ratio()
    ta, tb = TOKEN.findall(a), TOKEN.findall(b)
    token = difflib.SequenceMatcher(None, ta, tb).ratio()
    return char, token


def _parser(language):
    module = __import__(GRAMMARS[language])
    return Parser(Language(module.language()))


def _has_error(node):
    if node.has_error:
        return True
    for child in node.children:
        if _has_error(child):
            return True
    return False


def syntax_valid(language, text):
    try:
        tree = _parser(language).parse(text.encode("utf-8"))
        return not _has_error(tree.root_node)
    except Exception:
        return False


def fragment_text(language, completion):
    if language == "python":
        return completion
    wrap = WRAP[language]
    return wrap[0] + completion + wrap[1]


def check_syntax(language, completion, prefix, suffix):
    fragment = syntax_valid(language, fragment_text(language, completion))
    merged = syntax_valid(language, prefix + completion + suffix)
    return fragment, merged
