"""
sink.py

What the engines talk to. NullSink does nothing (the default, and the only one when no database is configured). PostgresSink builds the
rows of each step (rows.StepRecorder) on the control loop's thread and hands them to the writer thread; nothing here can raise into the loop.
"""

from src import config
from src.persistence.rows import StepRecorder


class NullSink:
    enabled = False

    def __init__(self, reason="no database configured (AUTOSCALER_DATABASE_URL is empty)"):
        self.reason = reason

    def record_step(self, payload, extras=None):
        pass

    def status(self):
        return {"enabled": False, "state": "off", "reason": self.reason}

    def close(self):
        pass


class PostgresSink:
    enabled = True

    def __init__(self, writer, recorder=None):
        self.writer = writer
        self.recorder = recorder or StepRecorder(config.CRITICAL_PATH_SOURCE, config.CRITICAL_PATH_SNAPSHOT_EVERY, config.STORE_FRAMES)
        self.errors = 0

    def record_step(self, payload, extras=None):
        try:
            batch = self.recorder.rows(payload, extras)
            self.writer.put(batch)
        except Exception as e:                       # a malformed payload must never reach the control loop
            self.errors += 1
            self.writer.last_error = f"row build failed: {type(e).__name__}: {e}"

    def status(self):
        s = self.writer.status()
        s["row_errors"] = self.errors
        return s

    def close(self):
        self.writer.close()
