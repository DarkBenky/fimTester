import os

EXTENSIONS = {
    ".c": "c",
    ".h": "c",
    ".go": "go",
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".jsx": "javascript",
}

SKIP_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode", ".venv", "venv", "node_modules",
    "vendor", "build", "dist", "target", "__pycache__", ".next", ".cache",
}


def collect_files(root, languages):
    found = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            language = EXTENSIONS.get(ext)
            if language is None or language not in languages:
                continue
            path = os.path.join(dirpath, filename)
            if not os.path.isfile(path):
                continue
            found[os.path.normpath(path)] = language
    return found


def read_source(path):
    with open(path, "rb") as f:
        raw = f.read()
    if b"\x00" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
