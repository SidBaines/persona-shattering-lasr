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

    def _render_lines(
        self, record: dict[str, Any], width: int
    ) -> list[tuple[str, str | None, bool]]:
        json_str = json.dumps(record, indent=2, ensure_ascii=False)
        raw_lines = json_str.splitlines()
        wrapped: list[tuple[str, str | None, bool]] = []
        current_key: str | None = None
        for line in raw_lines:
            is_key_line = self._is_key_line(line)
            if is_key_line:
                current_key = self._extract_key(line)
            if not line:
                wrapped.append(("", current_key, is_key_line))
                continue
            wrapped_lines = textwrap.wrap(
                line,
                width=width,
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
            for idx, wrapped_line in enumerate(wrapped_lines):
                wrapped.append((wrapped_line, current_key, is_key_line if idx == 0 else False))
        return wrapped

    def _main(self, stdscr: curses.window) -> None:
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)
        curses.use_default_colors()
        self._init_colors()

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
            self._add_colored(stdscr, 0, 0, header, self.COLOR_HEADER, width)

            visible = record_lines[self.line_offset : self.line_offset + body_height]
            for idx, (line, field_name, is_key_line) in enumerate(visible, start=1):
                if is_key_line and field_name:
                    self._add_key_value_line(stdscr, idx, line, field_name, width)
                else:
                    style = self._style_for_field(field_name, is_key_line)
                    self._add_colored(stdscr, idx, 0, line, style, width)

            footer = (
                "Up/Down: scroll  n/p: prev/next  PgUp/PgDn: page  "
                "g/G: first/last  q: quit"
            )
            self._add_colored(stdscr, height - 1, 0, footer, self.COLOR_FOOTER, width)
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

    def _init_colors(self) -> None:
        if not curses.has_colors():
            self.COLOR_HEADER = 0
            self.COLOR_FOOTER = 0
            self._field_pair_map = {}
            self._field_color_ids = {}
            return
        curses.start_color()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)
        self.COLOR_HEADER = curses.color_pair(1) | curses.A_BOLD
        self.COLOR_FOOTER = curses.color_pair(2)
        self._header_palette = [
            curses.COLOR_CYAN,
            curses.COLOR_YELLOW,
            curses.COLOR_GREEN,
            curses.COLOR_MAGENTA,
            curses.COLOR_BLUE,
            curses.COLOR_RED,
        ]
        self._content_color = curses.COLOR_WHITE
        self._field_pair_map: dict[str, tuple[int, int]] = {}
        self._next_field_pair = 10

    def _add_colored(
        self,
        stdscr: curses.window,
        y: int,
        x: int,
        text: str,
        style: int,
        width: int,
    ) -> None:
        stdscr.addnstr(y, x, text, max(0, width - 1), style)

    def _add_key_value_line(
        self,
        stdscr: curses.window,
        y: int,
        line: str,
        field_name: str,
        width: int,
    ) -> None:
        colon_index = line.find('":')
        if colon_index == -1:
            style = self._style_for_field(field_name, True)
            self._add_colored(stdscr, y, 0, line, style, width)
            return
        key_part = line[: colon_index + 2]
        value_part = line[colon_index + 2 :]
        header_style = self._style_for_field(field_name, True)
        content_style = self._style_for_field(field_name, False)
        self._add_colored(stdscr, y, 0, key_part, header_style, width)
        self._add_colored(
            stdscr,
            y,
            min(len(key_part), max(0, width - 1)),
            value_part,
            content_style,
            width - len(key_part),
        )

    def _style_for_field(self, field_name: str | None, is_key_line: bool) -> int:
        if not field_name:
            return 0
        header_pair, content_pair = self._get_field_pairs(field_name)
        base = curses.color_pair(header_pair if is_key_line else content_pair)
        if is_key_line:
            return base | curses.A_BOLD
        return base

    def _get_field_pairs(self, field_name: str) -> tuple[int, int]:
        if field_name in self._field_pair_map:
            return self._field_pair_map[field_name]
        if not curses.has_colors():
            self._field_pair_map[field_name] = (0, 0)
            return (0, 0)
        index = len(self._field_pair_map)
        header_color = self._header_palette[index % len(self._header_palette)]
        content_color = self._content_color
        header_pair = self._next_field_pair
        content_pair = self._next_field_pair + 1
        curses.init_pair(header_pair, header_color, -1)
        curses.init_pair(content_pair, content_color, -1)
        self._field_pair_map[field_name] = (header_pair, content_pair)
        self._next_field_pair += 2
        return (header_pair, content_pair)

    def _is_key_line(self, line: str) -> bool:
        stripped = line.lstrip()
        return stripped.startswith('"') and '":' in stripped

    def _extract_key(self, line: str) -> str | None:
        stripped = line.lstrip()
        if not self._is_key_line(stripped):
            return None
        end_quote = stripped.find('":')
        if end_quote == -1:
            return None
        return stripped[1:end_quote]
