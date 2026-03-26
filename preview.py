#!/usr/bin/env python3
"""Preview the blog post with auto-rebuild on file changes."""

import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

BLOG_DIR = Path.home() / "projects/blog/main"
CONTENT_DIR = Path("content")
OUTPUT_DIR = Path("output")
PORT = 8000

sys.path.insert(0, str(BLOG_DIR))
import build

build.CONTENT_DIR = CONTENT_DIR
build.OUTPUT_DIR = OUTPUT_DIR
build.TEMPLATE_DIR = BLOG_DIR
build.STATIC_DIRS = []


def do_build():
    # Reload build module so it picks up any changes to build.py
    try:
        build.build()
    except Exception as e:
        print(f"\n*** Build error: {e} ***\n")


def serve():
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(OUTPUT_DIR), **kw)
        def log_message(self, fmt, *args):
            pass  # quiet

    httpd = HTTPServer(("", PORT), Handler)
    print(f"Serving at http://localhost:{PORT}/conditional-poisson-sampling/")
    httpd.serve_forever()


def watch():
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    last_build = [0]

    class Rebuild(FileSystemEventHandler):
        def on_any_event(self, event):
            if event.src_path and '/output' in event.src_path:
                return
            if event.src_path and '/.ipynb_checkpoints' in event.src_path:
                return
            now = time.time()
            if now - last_build[0] < 2:  # debounce
                return
            last_build[0] = now
            print(f"\n--- Rebuilding ({Path(event.src_path).name} changed) ---")
            do_build()

    observer = Observer()
    observer.schedule(Rebuild(), str(CONTENT_DIR), recursive=True)
    observer.schedule(Rebuild(), str(BLOG_DIR / "template.html"), recursive=False)
    observer.start()
    return observer


if __name__ == "__main__":
    do_build()
    observer = watch()
    try:
        serve()
    except KeyboardInterrupt:
        observer.stop()
        observer.join()
