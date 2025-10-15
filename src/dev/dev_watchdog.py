import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import sys
import os

class RestartHandler(FileSystemEventHandler):
    def __init__(self, run_cmd):
        self.run_cmd = run_cmd
        self.proc = None
        self.restart()

    def restart(self):
        if self.proc:
            self.proc.terminate()
        self.proc = subprocess.Popen(self.run_cmd)

    def on_any_event(self, event):
        if event.src_path.endswith('.py'):
            print("Änderung erkannt, Anwendung wird neu gestartet...")
            self.restart()

if __name__ == "__main__":
    run_cmd = [sys.executable, "-m", "src.gui.main"]
    event_handler = RestartHandler(run_cmd)
    observer = Observer()
    observer.schedule(event_handler, path="src/", recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
