#!/usr/bin/env python3
"""Extract function snippets from source files and generate popover.js.

Run at build time (via [tool.blog] preprocess) to keep pill links fresh.
Reads Python and R source files, extracts function bodies, syntax-highlights
them with Pygments, and writes content/figures/popover.js with inline JSON
data and popover UI code.
"""

import ast
import json
from pathlib import Path

from pygments import highlight
from pygments.lexers import PythonLexer, SLexer
from pygments.formatters import HtmlFormatter

REPO = "timvieira/conditional-poisson-sampling"
BRANCH = "main"
GITHUB_BASE = f"https://github.com/{REPO}/blob/{BRANCH}"

PYTHON_FILES = [
    "test_identities.py",
    "bench_timing.py",
    "bench_samplers.py",
    "torch_fft_prototype.py",
    "conditional_poisson_numpy.py",
]

R_FILES = ["bench_timing_r.R"]

OUTPUT = Path("content/figures/popover.js")


def extract_python_functions(filepath):
    """Extract all function definitions from a Python file using the AST."""
    source = Path(filepath).read_text()
    lines = source.splitlines()
    tree = ast.parse(source, filename=filepath)
    snippets = {}

    # Top-level functions
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            end = node.end_lineno
            body = "\n".join(lines[start - 1 : end])
            snippets[node.name] = (start, end, body)

    # Class methods
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = item.decorator_list[0].lineno if item.decorator_list else item.lineno
                    end = item.end_lineno
                    body = "\n".join(lines[start - 1 : end])
                    key = f"{node.name}.{item.name}"
                    snippets[key] = (start, end, body)

    return snippets


def extract_r_blocks(filepath):
    """Extract labeled code blocks from the R benchmark file.

    The R file is structured as comment-delimited blocks, each containing
    a time_it() call. We extract blocks by matching the function names
    referenced in pill titles.
    """
    source = Path(filepath).read_text()
    lines = source.splitlines()
    snippets = {}

    # Extract blocks between "# ---" comment headers.
    # Each block runs from its header through the emit() call.
    blocks = []
    current_start = None
    for i, line in enumerate(lines, 1):
        if line.startswith("# ---"):
            if current_start is not None:
                blocks.append((current_start, i - 1))
            current_start = i
    if current_start is not None:
        blocks.append((current_start, len(lines)))

    # Map pill title keywords to blocks
    targets = {
        "UPMEqfromw + UPMEpikfromq": ["UPMEqfromw", "UPMEpikfromq"],
        "UPmaxentropy": ["UPmaxentropy"],
    }

    for title, keywords in targets.items():
        for block_start, block_end in blocks:
            block_text = "\n".join(lines[block_start - 1 : block_end])
            if all(kw in block_text for kw in keywords):
                snippets[title] = (block_start, block_end, block_text)
                break

    return snippets


def extract_module_docstring(filepath):
    """Extract the module-level docstring from a Python file."""
    source = Path(filepath).read_text()
    tree = ast.parse(source, filename=filepath)
    docstring = ast.get_docstring(tree)
    return docstring or Path(filepath).name


def highlight_python(source):
    return highlight(source, PythonLexer(), HtmlFormatter(nowrap=True))


def highlight_r(source):
    return highlight(source, SLexer(), HtmlFormatter(nowrap=True))


def build_snippet_data():
    """Build the complete snippet dictionary."""
    data = {}

    for filepath in PYTHON_FILES:
        if not Path(filepath).exists():
            print(f"  warning: {filepath} not found, skipping")
            continue
        funcs = extract_python_functions(filepath)
        for name, (start, end, source) in funcs.items():
            key = f"{filepath}::{name}"
            data[key] = {
                "file": filepath,
                "name": name,
                "start": start,
                "end": end,
                "html": highlight_python(source),
            }
        # Also store a file-level entry with the docstring
        docstring = extract_module_docstring(filepath)
        data[f"{filepath}::__file__"] = {
            "file": filepath,
            "name": filepath,
            "start": 1,
            "end": 1,
            "html": highlight_python(f'"""{docstring}"""') if docstring else "",
        }

    for filepath in R_FILES:
        if not Path(filepath).exists():
            print(f"  warning: {filepath} not found, skipping")
            continue
        blocks = extract_r_blocks(filepath)
        for name, (start, end, source) in blocks.items():
            key = f"{filepath}::{name}"
            data[key] = {
                "file": filepath,
                "name": name,
                "start": start,
                "end": end,
                "html": highlight_r(source),
            }

    return data


POPOVER_CSS = r"""
.code-popover {
    position: absolute;
    z-index: 1000;
    background: #f6f1ea;
    border: 1px solid #d0d0d0;
    border-radius: 6px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    max-width: min(720px, 90vw);
    max-height: 60vh;
    overflow-y: auto;
    font-size: 13px;
    line-height: 1.5;
}
.code-popover .popover-header {
    padding: 6px 14px;
    border-bottom: 1px solid #eee;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: system-ui, -apple-system, sans-serif;
    font-size: 12px;
    color: #666;
    background: #efe9e0;
    position: sticky;
    top: 0;
    z-index: 1;
    cursor: grab;
    user-select: none;
}
.code-popover .popover-header:active {
    cursor: grabbing;
}
.code-popover .popover-header .fn-name {
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    color: #333;
    font-weight: 600;
}
.code-popover .popover-header a {
    color: #0366d6;
    text-decoration: none;
    white-space: nowrap;
    margin-left: 16px;
}
.code-popover .popover-header a:hover {
    text-decoration: underline;
}
.code-popover .popover-body {
    overflow-x: auto;
    padding: 10px 14px;
}
.code-popover .popover-body pre {
    margin: 0;
    white-space: pre;
    overflow-x: auto;
    font-size: 12px;
    line-height: 1.45;
}
.code-popover .popover-separator {
    border: none;
    border-top: 1px solid #eee;
    margin: 0;
}
@media (max-width: 600px) {
    .code-popover {
        position: fixed !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        top: auto !important;
        max-width: 100vw;
        max-height: 50vh;
        border-radius: 12px 12px 0 0;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.2);
    }
}
"""

POPOVER_JS = r"""
(function() {
    var GITHUB_BASE = '""" + GITHUB_BASE + r"""';
    var activePopover = null;

    function dismiss() {
        if (activePopover) {
            activePopover.remove();
            activePopover = null;
        }
    }

    function positionPopover(popover, anchor) {
        // Temporarily make visible to measure
        popover.style.visibility = 'hidden';
        popover.style.display = 'block';
        document.body.appendChild(popover);

        var rect = anchor.getBoundingClientRect();
        var popRect = popover.getBoundingClientRect();
        var scrollX = window.pageXOffset;
        var scrollY = window.pageYOffset;

        // Position below the pill
        var top = rect.bottom + scrollY + 6;
        var left = rect.left + scrollX + rect.width / 2 - popRect.width / 2;

        // Keep within viewport
        if (left < 8) left = 8;
        if (left + popRect.width > document.documentElement.clientWidth - 8) {
            left = document.documentElement.clientWidth - popRect.width - 8;
        }

        // If it would go below viewport, show above instead
        if (rect.bottom + popRect.height + 6 > window.innerHeight) {
            top = rect.top + scrollY - popRect.height - 6;
        }

        popover.style.top = top + 'px';
        popover.style.left = left + 'px';
        popover.style.visibility = '';
    }

    function makeDraggable(popover, handle) {
        var startX, startY, origLeft, origTop;
        handle.addEventListener('mousedown', function(e) {
            if (e.target.tagName === 'A') return;  // don't drag when clicking links
            e.preventDefault();
            startX = e.clientX;
            startY = e.clientY;
            origLeft = popover.offsetLeft;
            origTop = popover.offsetTop;
            function onMove(e) {
                popover.style.left = (origLeft + e.clientX - startX) + 'px';
                popover.style.top = (origTop + e.clientY - startY) + 'px';
            }
            function onUp() {
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', onUp);
            }
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
        });
    }

    function createPopover(snippets) {
        var popover = document.createElement('div');
        popover.className = 'code-popover';

        for (var i = 0; i < snippets.length; i++) {
            var s = snippets[i];
            if (i > 0) {
                var sep = document.createElement('hr');
                sep.className = 'popover-separator';
                popover.appendChild(sep);
            }

            var header = document.createElement('div');
            header.className = 'popover-header';

            var nameSpan = document.createElement('span');
            nameSpan.className = 'fn-name';
            nameSpan.textContent = s.name;
            header.appendChild(nameSpan);

            var link = document.createElement('a');
            link.href = GITHUB_BASE + '/' + s.file + '#L' + s.start + '-L' + s.end;
            link.target = '_blank';
            link.textContent = 'View on GitHub \u2192';
            header.appendChild(link);

            popover.appendChild(header);

            var body = document.createElement('div');
            body.className = 'popover-body highlight';
            var pre = document.createElement('pre');
            pre.innerHTML = s.html;
            body.appendChild(pre);
            popover.appendChild(body);
        }

        makeDraggable(popover, popover.querySelector('.popover-header'));
        return popover;
    }

    function findSnippets(pill) {
        var href = pill.getAttribute('href') || '';
        var title = pill.getAttribute('title') || '';

        // Extract filename from href
        var file = href.split('#')[0];
        // Remove any URL prefix if present
        file = file.split('/').pop();

        if (!title && !href.includes('#')) {
            // File-level link: show module docstring
            var key = file + '::__file__';
            if (SNIPPET_DATA[key]) {
                return [SNIPPET_DATA[key]];
            }
            return [];
        }

        // Parse function names from title (comma-separated)
        var names = title ? title.split(',').map(function(s) { return s.trim(); }) : [];

        // Fallback: use anchor from href
        if (names.length === 0) {
            var anchor = href.split('#')[1];
            if (anchor) names = [anchor];
        }

        var results = [];
        for (var i = 0; i < names.length; i++) {
            var key = file + '::' + names[i];
            if (SNIPPET_DATA[key]) {
                results.push(SNIPPET_DATA[key]);
            }
        }
        return results;
    }

    document.addEventListener('DOMContentLoaded', function() {
        var pills = document.querySelectorAll('a.verified');

        pills.forEach(function(pill) {
            pill.style.cursor = 'pointer';
            pill.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();

                // If clicking the same pill, just dismiss
                if (activePopover && activePopover._pill === pill) {
                    dismiss();
                    return;
                }

                dismiss();

                var snippets = findSnippets(pill);
                if (snippets.length === 0) {
                    // Fallback: open on GitHub
                    var href = pill.getAttribute('href');
                    if (href) window.open(GITHUB_BASE + '/' + href, '_blank');
                    return;
                }

                var popover = createPopover(snippets);
                popover._pill = pill;
                activePopover = popover;
                positionPopover(popover, pill);
            });
        });

        document.addEventListener('click', function(e) {
            if (activePopover && !activePopover.contains(e.target) && !e.target.closest('a.verified')) {
                dismiss();
            }
        });

        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') dismiss();
        });
    });
})();
"""


def main():
    print("Extracting snippets...")
    data = build_snippet_data()
    print(f"  {len(data)} snippets extracted")

    # Build the output JS file
    js = f"var SNIPPET_DATA = {json.dumps(data)};\n"
    js += POPOVER_JS

    # Wrap CSS injection into the JS
    css_js = (
        "var _style = document.createElement('style');\n"
        f"_style.textContent = {json.dumps(POPOVER_CSS)};\n"
        "document.head.appendChild(_style);\n"
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(css_js + js)
    print(f"  wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
