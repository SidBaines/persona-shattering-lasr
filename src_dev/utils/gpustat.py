#!/usr/bin/env python3

import argparse
import csv
import curses
from dataclasses import dataclass
import io
import shutil
import subprocess
import time

# Ranges for color coding (in percentages)
BLUE_RANGE = (-1, 0)
GREEN_RANGE = (0, 40)
ORANGE_RANGE = (40, 75)
RED_RANGE = (75, 100)

# Temperature range (in Celsius)
TEMP_MIN = 20
TEMP_MAX = 100

# Color pair numbers
BLUE_PAIR = 1
GREEN_PAIR = 2
YELLOW_PAIR = 3
RED_PAIR = 4

NVIDIA_BACKEND = "nvidia"
AMD_BACKEND = "amd"


@dataclass
class GPUInfo:
    index: str
    name: str
    temp: str = "N/A"
    fan: str = "N/A"
    power: str = "N/A"
    used_mem: str = "N/A"
    total_mem: str = "N/A"
    util: str = "N/A"
    vendor: str = "unknown"


def setup_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(BLUE_PAIR, curses.COLOR_BLUE, -1)
    curses.init_pair(GREEN_PAIR, curses.COLOR_GREEN, -1)
    curses.init_pair(YELLOW_PAIR, curses.COLOR_YELLOW, -1)
    curses.init_pair(RED_PAIR, curses.COLOR_RED, -1)


def get_color_pair(value, max_value, is_temperature=False):
    if value in {"", "N/A", None} or max_value in {"", "N/A", None}:
        return 0

    if is_temperature:
        percentage = (float(value) - TEMP_MIN) / (TEMP_MAX - TEMP_MIN) * 100
    else:
        try:
            percentage = (float(value) / float(max_value)) * 100
        except (ValueError, ZeroDivisionError):
            return 0

    if BLUE_RANGE[0] <= percentage <= BLUE_RANGE[1]:
        return curses.color_pair(BLUE_PAIR)
    if GREEN_RANGE[0] < percentage <= GREEN_RANGE[1]:
        return curses.color_pair(GREEN_PAIR)
    if ORANGE_RANGE[0] < percentage <= ORANGE_RANGE[1]:
        return curses.color_pair(YELLOW_PAIR)
    return curses.color_pair(RED_PAIR)


def detect_backend():
    if shutil.which("nvidia-smi"):
        return NVIDIA_BACKEND
    if shutil.which("rocm-smi"):
        return AMD_BACKEND
    raise RuntimeError("No supported GPU status command found. Expected `nvidia-smi` or `rocm-smi`.")


def run_command(cmd):
    return subprocess.check_output(cmd, text=True).strip()


def read_csv_output(cmd):
    output = run_command(cmd)
    if not output:
        return []
    return list(csv.DictReader(io.StringIO(output)))


def parse_numeric_field(value):
    if value is None:
        return "N/A"
    value = value.strip()
    if not value or value.upper() == "N/A":
        return "N/A"
    return value


def bytes_to_mib(value):
    numeric = parse_numeric_field(value)
    if numeric == "N/A":
        return "N/A"
    return f"{int(float(numeric) / (1024 ** 2))}"


def get_nvidia_gpu_info():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,temperature.gpu,fan.speed,power.draw,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = run_command(cmd)
    gpus = []
    for line in output.splitlines():
        index, name, temp, fan, power, used_mem, total_mem, util = [part.strip() for part in line.split(",")]
        gpus.append(
            GPUInfo(
                index=index,
                name=name,
                temp=parse_numeric_field(temp),
                fan=parse_numeric_field(fan),
                power=parse_numeric_field(power),
                used_mem=parse_numeric_field(used_mem),
                total_mem=parse_numeric_field(total_mem),
                util=parse_numeric_field(util),
                vendor=NVIDIA_BACKEND,
            )
        )
    return gpus


def get_amd_gpu_info():
    status_rows = read_csv_output(
        [
            "rocm-smi",
            "--showproductname",
            "--showtemp",
            "--showfan",
            "--showpower",
            "--showmemuse",
            "--showuse",
            "--showmeminfo",
            "vram",
            "--csv",
        ]
    )
    if not status_rows:
        return []

    gpus = []
    for row in status_rows:
        name = row.get("Card series") or row.get("Card model") or "AMD GPU"
        gpus.append(
            GPUInfo(
                index=row.get("device", "").removeprefix("card") or "?",
                name=name,
                temp=parse_numeric_field(
                    row.get("Temperature (Sensor junction) (C)")
                    or row.get("Temperature (Sensor edge) (C)")
                    or row.get("Temperature (Sensor memory) (C)")
                ),
                fan=parse_numeric_field(row.get("Fan speed (%)")),
                power=parse_numeric_field(row.get("Average Graphics Package Power (W)")),
                used_mem=bytes_to_mib(row.get("VRAM Total Used Memory (B)")),
                total_mem=bytes_to_mib(row.get("VRAM Total Memory (B)")),
                util=parse_numeric_field(row.get("GPU use (%)")),
                vendor=AMD_BACKEND,
            )
        )
    return gpus


def get_gpu_info():
    backend = detect_backend()
    if backend == NVIDIA_BACKEND:
        return get_nvidia_gpu_info()
    return get_amd_gpu_info()


def draw_metric(stdscr, y, label, value, suffix="", color=0):
    stdscr.addstr(y, 0, f"{label} : ")
    stdscr.addstr(f"{value}{suffix}", color)


def display_gpu_info(stdscr):
    while True:
        gpu_info = get_gpu_info()
        stdscr.clear()
        for i, gpu in enumerate(gpu_info):
            y_offset = i * 8
            stdscr.addstr(y_offset, 0, f"GPU {gpu.index}")
            stdscr.addstr(y_offset + 1, 0, "=======")
            stdscr.addstr(y_offset + 2, 0, f"Name  : {gpu.name}")

            draw_metric(
                stdscr,
                y_offset + 3,
                "Temp ",
                gpu.temp,
                "°C" if gpu.temp != "N/A" else "",
                get_color_pair(gpu.temp, TEMP_MAX, is_temperature=True),
            )
            draw_metric(
                stdscr,
                y_offset + 4,
                "Fan  ",
                gpu.fan,
                "%" if gpu.fan != "N/A" else "",
                get_color_pair(gpu.fan, 100),
            )
            draw_metric(
                stdscr,
                y_offset + 5,
                "Pwr  ",
                gpu.power,
                "W" if gpu.power != "N/A" else "",
                get_color_pair(gpu.power, 300),
            )
            draw_metric(
                stdscr,
                y_offset + 6,
                "Mem  ",
                (
                    f"{gpu.used_mem} / {gpu.total_mem} MiB"
                    if gpu.used_mem != "N/A" and gpu.total_mem != "N/A"
                    else "N/A"
                ),
                color=get_color_pair(gpu.used_mem, gpu.total_mem),
            )
            draw_metric(
                stdscr,
                y_offset + 7,
                "Util ",
                gpu.util,
                "%" if gpu.util != "N/A" else "",
                get_color_pair(gpu.util, 100),
            )

        stdscr.refresh()
        time.sleep(args.watch)


def print_gpu_info():
    for gpu in get_gpu_info():
        print(f"GPU {gpu.index}")
        print("=======")
        print(f"Name  : {gpu.name}")
        print(f"Temp  : {gpu.temp}{'°C' if gpu.temp != 'N/A' else ''}")
        print(f"Fan   : {gpu.fan}{'%' if gpu.fan != 'N/A' else ''}")
        print(f"Pwr   : {gpu.power}{'W' if gpu.power != 'N/A' else ''}")
        if gpu.used_mem != "N/A" and gpu.total_mem != "N/A":
            print(f"Mem   : {gpu.used_mem} / {gpu.total_mem} MiB")
        else:
            print("Mem   : N/A")
        print(f"Util  : {gpu.util}{'%' if gpu.util != 'N/A' else ''}")
        print()


def main(stdscr):
    global args
    curses.curs_set(0)
    setup_colors()
    stdscr.nodelay(1)

    try:
        display_gpu_info(stdscr)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display GPU information")
    parser.add_argument("--watch", type=float, default=1.0, help="Refresh rate in seconds (can be fractional)")
    args = parser.parse_args()

    if args.watch:
        curses.wrapper(main)
    else:
        print_gpu_info()
