"""Curses-based TUI for browsing JSONL records."""

from __future__ import annotations

import curses
import json
import textwrap
from pathlib import Path
from typing import Any

from scripts.utils import read_jsonl


class JsonlViewer:
    """Simple TUI for scrolling JSONL records."""

    def __init__(self, path: str | Path, start_index: int = 0) -> None:
        self.path = Path(path)
        self.records: list[dict[str, Any]] = read_jsonl(self.path)
        self.index = max(0, min(start_index, len(self.records) - 1)) if self.records else 0
        self.line_offset = 0

    def run(self) -> None:
        """Run the TUI viewer."""
        curses.wrapper(self._main)

    def _render_lines(self, record: dict[str, Any], width: int) -> list[str]:
        json_str = json.dumps(record, indent=2, ensure_ascii=False)
        raw_lines = json_str.splitlines()
        wrapped: list[str] = []
        for line in raw_lines:
            if not line:
                wrapped.append("")
                continue
            wrapped.extend(
                textwrap.wrap(
                    line,
                    width=width,
                    replace_whitespace=False,
                    drop_whitespace=False,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
                or [""]
            )
        return wrapped

    def _main(self, stdscr: curses.window) -> None:
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)
        curses.use_default_colors()

        while True:
            height, width = stdscr.getmaxyx()
            body_height = max(1, height - 2)
            stdscr.erase()

            if not self.records:
                msg = f"No records found in {self.path}"
                stdscr.addnstr(0, 0, msg, max(0, width - 1))
                stdscr.addnstr(
                    height - 1,
                    0,
                    "q: quit",
                    max(0, width - 1),
                )
                stdscr.refresh()
                key = stdscr.getch()
                if key in (ord("q"), 27):
                    break
                continue

            record_lines = self._render_lines(self.records[self.index], max(10, width - 1))
            max_offset = max(0, len(record_lines) - body_height)
            self.line_offset = min(self.line_offset, max_offset)

            header = (
                f"{self.path}  Record {self.index + 1}/{len(self.records)}"
                f"  Line {self.line_offset + 1}/{max(1, len(record_lines))}"
            )
            stdscr.addnstr(0, 0, header, max(0, width - 1))

            visible = record_lines[self.line_offset : self.line_offset + body_height]
            for idx, line in enumerate(visible, start=1):
                stdscr.addnstr(idx, 0, line, max(0, width - 1))

            footer = (
                "Up/Down: scroll  n/p: prev/next  PgUp/PgDn: page  "
                "g/G: first/last  q: quit"
            )
            stdscr.addnstr(height - 1, 0, footer, max(0, width - 1))
            stdscr.refresh()

            key = stdscr.getch()
            if key in (ord("q"), 27):
                break
            if key in (curses.KEY_DOWN, ord("j")):
                if self.line_offset < max_offset:
                    self.line_offset += 1
                elif self.index < len(self.records) - 1:
                    self.index += 1
                    self.line_offset = 0
                continue
            if key in (curses.KEY_UP, ord("k")):
                if self.line_offset > 0:
                    self.line_offset -= 1
                elif self.index > 0:
                    self.index -= 1
                    prev_lines = self._render_lines(
                        self.records[self.index], max(10, width - 1)
                    )
                    self.line_offset = max(0, len(prev_lines) - body_height)
                continue
            if key in (curses.KEY_NPAGE,):
                self.line_offset = min(max_offset, self.line_offset + body_height)
                continue
            if key in (curses.KEY_PPAGE,):
                self.line_offset = max(0, self.line_offset - body_height)
                continue
            if key in (ord("n"), curses.KEY_RIGHT):
                if self.index < len(self.records) - 1:
                    self.index += 1
                    self.line_offset = 0
                continue
            if key in (ord("p"), curses.KEY_LEFT):
                if self.index > 0:
                    self.index -= 1
                    self.line_offset = 0
                continue
            if key == ord("g"):
                self.index = 0
                self.line_offset = 0
                continue
            if key == ord("G"):
                self.index = len(self.records) - 1
                self.line_offset = 0
                continue
