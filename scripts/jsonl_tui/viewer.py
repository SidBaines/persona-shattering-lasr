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

    def __init__(
        self,
        path: str | Path,
        start_index: int = 0,
        variant_fields: list[str] | None = None,
        conversation_field: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.records: list[dict[str, Any]] = read_jsonl(self.path)
        self.variant_fields = variant_fields
        self.conversation_field = conversation_field
        self.current_variant_index: int = 0
        self._build_groups(start_index)
        self.line_offset = 0

    def _grouped_variant_mode(self) -> bool:
        """Whether variant-fields mode should group by question and show all fields."""
        return self.variant_fields is not None and "question" in self.variant_fields

    def _build_groups(self, start_index: int) -> None:
        self.question_labels: list[str] = []
        self.grouped_records: list[list[dict[str, Any]]] = []
        if not self.records:
            self.question_index = 0
            self.response_index = 0
            return

        if self.variant_fields is not None:
            if self._grouped_variant_mode():
                groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
                for idx, record in enumerate(self.records):
                    question = record.get("question")
                    if question is None:
                        question = f"Record {idx + 1}"
                    groups.setdefault(question, []).append((idx, record))

                for question, items in groups.items():
                    def sort_key(item: tuple[int, dict[str, Any]]) -> tuple[int, int, int]:
                        original_index, record = item
                        response_index = record.get("response_index")
                        if isinstance(response_index, int):
                            return (0, response_index, original_index)
                        return (1, original_index, original_index)

                    items_sorted = sorted(items, key=sort_key)
                    self.question_labels.append(question)
                    self.grouped_records.append([record for _, record in items_sorted])
            else:
                # In variant-fields mode: one group per record, no question-based grouping.
                for idx, record in enumerate(self.records):
                    label = record.get("question") or f"Record {idx + 1}"
                    self.question_labels.append(label)
                    self.grouped_records.append([record])
        else:
            groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
            for idx, record in enumerate(self.records):
                question = record.get("question")
                if question is None:
                    question = f"Record {idx + 1}"
                groups.setdefault(question, []).append((idx, record))

            for question, items in groups.items():
                def sort_key(item: tuple[int, dict[str, Any]]) -> tuple[int, int, int]:
                    original_index, record = item
                    response_index = record.get("response_index")
                    if isinstance(response_index, int):
                        return (0, response_index, original_index)
                    return (1, original_index, original_index)

                items_sorted = sorted(items, key=sort_key)
                self.question_labels.append(question)
                self.grouped_records.append([record for _, record in items_sorted])

        self.question_index = 0
        self.response_index = 0

        if 0 <= start_index < len(self.records):
            target = self.records[start_index]
            target_question = target.get("question")
            if target_question is None:
                target_question = f"Record {start_index + 1}"
            if target_question in self.question_labels:
                self.question_index = self.question_labels.index(target_question)
                if self.variant_fields is None:
                    try:
                        self.response_index = self.grouped_records[self.question_index].index(target)
                    except ValueError:
                        self.response_index = 0

    def run(self) -> None:
        """Run the TUI viewer."""
        curses.wrapper(self._main)

    def _scroll_first_navigation(self) -> bool:
        """Whether arrow-key navigation should scroll before changing records."""
        return (
            (self.variant_fields is not None and not self._grouped_variant_mode())
            or self.conversation_field is not None
        )

    def _next_group(self) -> None:
        if self.question_index < len(self.grouped_records) - 1:
            self.question_index += 1
            self.response_index = 0
            self.line_offset = 0

    def _prev_group(self) -> None:
        if self.question_index > 0:
            self.question_index -= 1
            self.response_index = 0
            self.line_offset = 0

    def _scroll_down(self, max_offset: int, amount: int = 1) -> None:
        self.line_offset = min(max_offset, self.line_offset + amount)

    def _scroll_up(self, amount: int = 1) -> None:
        self.line_offset = max(0, self.line_offset - amount)

    def _handle_key(
        self,
        key: int,
        *,
        max_offset: int,
        body_height: int,
        current_group: list[dict[str, Any]],
    ) -> bool:
        """Handle one keypress.

        Returns:
            True if the viewer should exit, else False.
        """
        if key in (ord("q"), 27):
            return True

        if key in (curses.KEY_DOWN,):
            if self._scroll_first_navigation():
                self._scroll_down(max_offset)
            else:
                self._next_group()
            return False

        if key in (curses.KEY_UP,):
            if self._scroll_first_navigation():
                self._scroll_up()
            else:
                self._prev_group()
            return False

        if key in (ord("j"),):
            self._scroll_down(max_offset)
            return False

        if key in (ord("k"),):
            self._scroll_up()
            return False

        if key in (curses.KEY_NPAGE,):
            self._scroll_down(max_offset, body_height)
            return False

        if key in (curses.KEY_PPAGE,):
            self._scroll_up(body_height)
            return False

        if key == ord("n"):
            self._next_group()
            return False

        if key == ord("p"):
            self._prev_group()
            return False

        if key in (ord("l"), curses.KEY_RIGHT):
            if self.variant_fields is not None:
                if self._grouped_variant_mode():
                    if self.response_index < len(current_group) - 1:
                        self.response_index += 1
                        self.line_offset = 0
                else:
                    self.current_variant_index = (
                        (self.current_variant_index + 1) % len(self.variant_fields)
                    )
                    self.line_offset = 0
            elif self.conversation_field is not None:
                self._next_group()
            elif self.response_index < len(current_group) - 1:
                self.response_index += 1
                self.line_offset = 0
            return False

        if key in (ord("h"), curses.KEY_LEFT):
            if self.variant_fields is not None:
                if self._grouped_variant_mode():
                    if self.response_index > 0:
                        self.response_index -= 1
                        self.line_offset = 0
                else:
                    self.current_variant_index = (
                        (self.current_variant_index - 1) % len(self.variant_fields)
                    )
                    self.line_offset = 0
            elif self.conversation_field is not None:
                self._prev_group()
            elif self.response_index > 0:
                self.response_index -= 1
                self.line_offset = 0
            return False

        if key == ord("g"):
            self.question_index = 0
            self.response_index = 0
            self.line_offset = 0
            return False

        if key == ord("G"):
            self.question_index = len(self.grouped_records) - 1
            self.response_index = 0
            self.line_offset = 0
            return False

        return False

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

    def _render_variant_lines(
        self, record: dict[str, Any], variant_field: str, width: int
    ) -> list[tuple[str, str | None, bool]]:
        """Render a clean prose view for variant-fields mode.

        Shows the question text followed by a separator and the selected variant's text.
        """
        question = record.get("question", "")
        variant_text = record.get(variant_field)
        if variant_text is None:
            variant_text = "(not available)"

        separator = "\u2550" * min(width - 1, 60)

        sections: list[tuple[str, str]] = [
            ("QUESTION", str(question)),
            (variant_field, str(variant_text)),
        ]

        lines: list[tuple[str, str | None, bool]] = []
        for section_label, section_text in sections:
            # Section header line
            lines.append((section_label, section_label, True))
            lines.append((separator, section_label, False))
            # Word-wrapped body
            for para in section_text.splitlines() or [""]:
                wrapped = textwrap.wrap(
                    para,
                    width=max(10, width - 1),
                    replace_whitespace=False,
                    drop_whitespace=False,
                    break_long_words=True,
                    break_on_hyphens=False,
                ) or [""]
                for wline in wrapped:
                    lines.append((wline, section_label, False))
            lines.append(("", None, False))
        return lines

    def _render_grouped_variant_lines(
        self, record: dict[str, Any], width: int
    ) -> list[tuple[str, str | None, bool]]:
        """Render all variant fields as sections for grouped variant mode."""
        fields = self.variant_fields or []
        separator = "\u2550" * min(width - 1, 60)

        lines: list[tuple[str, str | None, bool]] = []
        for field_name in fields:
            section_label = field_name.upper()
            value = record.get(field_name)
            if value is None:
                value_text = "(not available)"
            else:
                value_text = str(value)

            lines.append((section_label, section_label, True))
            lines.append((separator, section_label, False))
            for para in value_text.splitlines() or [""]:
                wrapped = textwrap.wrap(
                    para,
                    width=max(10, width - 1),
                    replace_whitespace=False,
                    drop_whitespace=False,
                    break_long_words=True,
                    break_on_hyphens=False,
                ) or [""]
                for wline in wrapped:
                    lines.append((wline, section_label, False))
            lines.append(("", None, False))
        return lines

    def _render_conversation_lines(
        self, record: dict[str, Any], conversation_field: str, width: int
    ) -> list[tuple[str, str | None, bool]]:
        """Render a conversation transcript from a list-valued field."""
        lines: list[tuple[str, str | None, bool]] = []
        transcript = record.get(conversation_field)
        if not isinstance(transcript, list):
            return [
                ("CONVERSATION", "CONVERSATION", True),
                ("(conversation field missing or not a list)", "CONVERSATION", False),
            ]

        summary_fields = (
            ("sample_id", record.get("sample_id")),
            ("editing_variant", record.get("editing_variant")),
            ("assistant_turn_count", record.get("assistant_turn_count")),
        )
        for key, value in summary_fields:
            if value is None:
                continue
            lines.append((key.upper(), key.upper(), True))
            for wrapped in textwrap.wrap(
                f"{value}",
                width=max(10, width - 1),
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]:
                lines.append((wrapped, key.upper(), False))
            lines.append(("", None, False))

        for idx, message in enumerate(transcript, start=1):
            if not isinstance(message, dict):
                continue
            role_raw = message.get("role", "unknown")
            if not isinstance(role_raw, str):
                role_raw = "unknown"
            role = role_raw.upper()
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)

            lines.append((f"{role} {idx}", role, True))
            for paragraph in content.splitlines() or [""]:
                wrapped = textwrap.wrap(
                    paragraph,
                    width=max(10, width - 1),
                    replace_whitespace=False,
                    drop_whitespace=False,
                    break_long_words=True,
                    break_on_hyphens=False,
                ) or [""]
                for wrapped_line in wrapped:
                    lines.append((wrapped_line, role, False))
            lines.append(("", None, False))

        if not lines:
            lines.append(("CONVERSATION", "CONVERSATION", True))
            lines.append(("(empty transcript)", "CONVERSATION", False))
        return lines

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

            current_group = self.grouped_records[self.question_index]
            self.response_index = min(self.response_index, len(current_group) - 1)
            current_record = current_group[self.response_index]

            if self.variant_fields is not None:
                if self._grouped_variant_mode():
                    record_lines = self._render_grouped_variant_lines(
                        current_record,
                        max(10, width - 1),
                    )
                    max_offset = max(0, len(record_lines) - body_height)
                    self.line_offset = min(self.line_offset, max_offset)
                    response_label = current_record.get("response_index")
                    response_count = len(current_group)
                    if isinstance(response_label, int):
                        response_display = (
                            f"{response_label} ({self.response_index + 1}/{response_count})"
                        )
                    else:
                        response_display = f"{self.response_index + 1}/{response_count}"
                    header = (
                        f"{self.path}  Group {self.question_index + 1}/{len(self.grouped_records)}"
                        f"  Sample {response_display}"
                        f"  Line {self.line_offset + 1}/{max(1, len(record_lines))}"
                    )
                    footer = (
                        "Up/Down: prev/next group  Left/Right: prev/next sample  "
                        "j/k: scroll  PgUp/PgDn: page  g/G: first/last  q: quit"
                    )
                else:
                    # Variant-fields mode: left/right cycles through fields.
                    num_variants = len(self.variant_fields)
                    self.current_variant_index = self.current_variant_index % max(1, num_variants)
                    current_field = self.variant_fields[self.current_variant_index]
                    record_lines = self._render_variant_lines(
                        current_record, current_field, max(10, width - 1)
                    )
                    max_offset = max(0, len(record_lines) - body_height)
                    self.line_offset = min(self.line_offset, max_offset)
                    variant_display = (
                        f"{current_field} ({self.current_variant_index + 1}/{num_variants})"
                    )
                    header = (
                        f"{self.path}  Question {self.question_index + 1}/{len(self.grouped_records)}"
                        f"  Variant: {variant_display}"
                        f"  Line {self.line_offset + 1}/{max(1, len(record_lines))}"
                    )
                    footer = (
                        "Up/Down or j/k: scroll  Left/Right or h/l: prev/next variant  "
                        "n/p: prev/next question  PgUp/PgDn: page  g/G: first/last  q: quit"
                    )
            elif self.conversation_field is not None:
                record_lines = self._render_conversation_lines(
                    current_record,
                    self.conversation_field,
                    max(10, width - 1),
                )
                max_offset = max(0, len(record_lines) - body_height)
                self.line_offset = min(self.line_offset, max_offset)
                header = (
                    f"{self.path}  Record {self.question_index + 1}/{len(self.grouped_records)}"
                    f"  Conversation field: {self.conversation_field}"
                    f"  Line {self.line_offset + 1}/{max(1, len(record_lines))}"
                )
                footer = (
                    "Up/Down or j/k: scroll  Left/Right or n/p: prev/next record  "
                    "PgUp/PgDn: page  g/G: first/last  q: quit"
                )
            else:
                record_lines = self._render_lines(current_record, max(10, width - 1))
                max_offset = max(0, len(record_lines) - body_height)
                self.line_offset = min(self.line_offset, max_offset)
                response_label = current_record.get("response_index")
                response_count = len(current_group)
                if isinstance(response_label, int):
                    response_display = (
                        f"{response_label} ({self.response_index + 1}/{response_count})"
                    )
                else:
                    response_display = f"{self.response_index + 1}/{response_count}"
                header = (
                    f"{self.path}  Question {self.question_index + 1}/{len(self.grouped_records)}"
                    f"  Response {response_display}"
                    f"  Line {self.line_offset + 1}/{max(1, len(record_lines))}"
                )
                footer = (
                    "Up/Down: prev/next question  Left/Right: prev/next response  "
                    "j/k: scroll  PgUp/PgDn: page  g/G: first/last  q: quit"
                )

            self._add_colored(stdscr, 0, 0, header, self.COLOR_HEADER, width)

            visible = record_lines[self.line_offset : self.line_offset + body_height]
            for idx, (line, field_name, is_key_line) in enumerate(visible, start=1):
                if is_key_line and field_name:
                    self._add_key_value_line(stdscr, idx, line, field_name, width)
                else:
                    style = self._style_for_field(field_name, is_key_line)
                    self._add_colored(stdscr, idx, 0, line, style, width)

            self._add_colored(stdscr, height - 1, 0, footer, self.COLOR_FOOTER, width)
            stdscr.refresh()

            key = stdscr.getch()
            if self._handle_key(
                key,
                max_offset=max_offset,
                body_height=body_height,
                current_group=current_group,
            ):
                break

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
