#!/usr/bin/env python3

import subprocess
import sys
import re
import time
import argparse
import curses

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

def setup_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(BLUE_PAIR, curses.COLOR_BLUE, -1)
    curses.init_pair(GREEN_PAIR, curses.COLOR_GREEN, -1)
    curses.init_pair(YELLOW_PAIR, curses.COLOR_YELLOW, -1)  # Using magenta as a stand-in for orange
    curses.init_pair(RED_PAIR, curses.COLOR_RED, -1)

def get_color_pair(value, max_value, is_temperature=False):
    if is_temperature:
        percentage = (float(value) - TEMP_MIN) / (TEMP_MAX - TEMP_MIN) * 100
    else:
        try:
            percentage = (float(value) / float(max_value)) * 100
        except (ValueError, ZeroDivisionError):
            return 0
    
    if BLUE_RANGE[0] <= percentage <= BLUE_RANGE[1]:
        return curses.color_pair(BLUE_PAIR)
    elif GREEN_RANGE[0] < percentage <= GREEN_RANGE[1]:
        return curses.color_pair(GREEN_PAIR)
    elif ORANGE_RANGE[0] < percentage <= ORANGE_RANGE[1]:
        return curses.color_pair(YELLOW_PAIR)
    else:
        return curses.color_pair(RED_PAIR)

def get_gpu_info():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,temperature.gpu,fan.speed,power.draw,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"
    ]
    output = subprocess.check_output(cmd).decode('utf-8').strip()
    return [line.split(',') for line in output.split('\n')]

def display_gpu_info(stdscr):
    while True:
        gpu_info = get_gpu_info()
        stdscr.clear()
        for i, gpu in enumerate(gpu_info):
            index, name, temp, fan, power, used_mem, total_mem, util = gpu
            y_offset = i * 8  # 8 lines per GPU
            stdscr.addstr(y_offset, 0, f"GPU {index}")
            stdscr.addstr(y_offset + 1, 0, "=======")
            stdscr.addstr(y_offset + 2, 0, f"Name  : {name}")
            
            stdscr.addstr(y_offset + 3, 0, f"Temp  : ")
            stdscr.addstr(f"{temp}°C", get_color_pair(temp, TEMP_MAX, is_temperature=True))
            
            stdscr.addstr(y_offset + 4, 0, f"Fan   : ")
            stdscr.addstr(f"{fan}%", get_color_pair(fan, 100))
            
            stdscr.addstr(y_offset + 5, 0, f"Pwr   : ")
            stdscr.addstr(f"{power}W", get_color_pair(power, 300))
            
            stdscr.addstr(y_offset + 6, 0, f"Mem   : ")
            stdscr.addstr(f"{used_mem} / {total_mem} MiB", get_color_pair(used_mem, total_mem))
            
            stdscr.addstr(y_offset + 7, 0, f"Util  : ")
            stdscr.addstr(f"{util}%", get_color_pair(util, 100))
        
        stdscr.refresh()
        time.sleep(args.watch)

def main(stdscr):
    global args
    curses.curs_set(0)  # Hide the cursor
    setup_colors()
    stdscr.nodelay(1)  # Make getch non-blocking
    
    try:
        display_gpu_info(stdscr)
    except KeyboardInterrupt:
        pass  # Allow clean exit with Ctrl+C

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display GPU information")
    parser.add_argument("--watch", type=float, default=1.0, help="Refresh rate in seconds (can be fractional)")
    args = parser.parse_args()

    if args.watch:
        curses.wrapper(main)
    else:
        gpu_info = get_gpu_info()
        for gpu in gpu_info:
            index, name, temp, fan, power, used_mem, total_mem, util = gpu
            print(f"GPU {index}")
            print("=======")
            print(f"Name  : {name}")
            print(f"Temp  : {temp}°C")
            print(f"Fan   : {fan}%")
            print(f"Pwr   : {power}W")
            print(f"Mem   : {used_mem} / {total_mem} MiB")
            print(f"Util  : {util}%")
            print()