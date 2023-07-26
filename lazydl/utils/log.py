import logging
import sys
import os
import inspect
import time


class Logger:
    COLORS = {
        'DEBUG': '\033[94m',
        'INFO': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[91m',
        'FILENAME': '\033[95m',
        'LINENO': '\033[95m',
    }
    RESET = '\033[0m'

    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.rank = 0

    def get_logger(self):
        return self.logger

    def set_rank(self, rank):
        self.rank = rank

    def format_with_color(self, message, color):
        return f'{color}{message}{self.RESET}'

    def debug(self, message):
        if self.rank == 0:
            frame = inspect.currentframe().f_back
            filename = os.path.basename(frame.f_code.co_filename)
            lineno = frame.f_lineno
            timestamp = time.strftime('%m-%d %H:%M:%S', time.localtime())
            formatted_time = self.format_with_color(timestamp, self.COLORS['DEBUG'])
            formatted_filename = self.format_with_color(filename, self.COLORS['FILENAME'])
            formatted_lineno = self.format_with_color(str(lineno), self.COLORS['LINENO'])
            print(f'\033[1m{self.COLORS["DEBUG"]}[DEBUG] {formatted_time}  {self.RESET} \033[1m[{formatted_filename}:\033[1m{formatted_lineno}] \033[1m{message}')

    def info(self, message):
        if self.rank == 0:
            frame = inspect.currentframe().f_back
            filename = os.path.basename(frame.f_code.co_filename)
            lineno = frame.f_lineno
            timestamp = time.strftime('%m-%d %H:%M:%S', time.localtime())
            formatted_time = self.format_with_color(timestamp, self.COLORS['INFO'])
            formatted_filename = self.format_with_color(filename, self.COLORS['FILENAME'])
            formatted_lineno = self.format_with_color(str(lineno), self.COLORS['LINENO'])
            print(f'\033[2m{formatted_time} {self.COLORS["INFO"]}{self.RESET} \033[2m[{formatted_filename}:\033[2m{formatted_lineno}] {message}')

    def warning(self, message):
        if self.rank == 0:
            frame = inspect.currentframe().f_back
            filename = os.path.basename(frame.f_code.co_filename)
            lineno = frame.f_lineno
            timestamp = time.strftime('%m-%d %H:%M:%S', time.localtime())
            formatted_time = self.format_with_color(timestamp, self.COLORS['WARNING'])
            formatted_filename = self.format_with_color(filename, self.COLORS['FILENAME'])
            formatted_lineno = self.format_with_color(str(lineno), self.COLORS['LINENO'])
            print(f'\033[1m{self.COLORS["WARNING"]}[WARNING] {formatted_time} {self.RESET} \033[1m[{formatted_filename}:\033[1m{formatted_lineno}] \033[1m{message}')

    def error(self, message):
        if self.rank == 0:
            frame = inspect.currentframe().f_back
            filename = os.path.basename(frame.f_code.co_filename)
            lineno = frame.f_lineno
            timestamp = time.strftime('%m-%d %H:%M:%S', time.localtime())
            formatted_time = self.format_with_color(timestamp, self.COLORS['ERROR'])
            formatted_filename = self.format_with_color(filename, self.COLORS['FILENAME'])
            formatted_lineno = self.format_with_color(str(lineno), self.COLORS['LINENO'])
            print(f'\033[1m{self.COLORS["ERROR"]}[ERROR] {formatted_time} {self.RESET} \033[1m[{formatted_filename}:\033[1m{formatted_lineno}] \033[1m{message}')

    def critical(self, message):
        if self.rank == 0:
            frame = inspect.currentframe().f_back
            filename = os.path.basename(frame.f_code.co_filename)
            lineno = frame.f_lineno
            timestamp = time.strftime('%m-%d %H:%M:%S', time.localtime())
            formatted_time = self.format_with_color(timestamp, self.COLORS['CRITICAL'])
            formatted_filename = self.format_with_color(filename, self.COLORS['FILENAME'])
            formatted_lineno = self.format_with_color(str(lineno), self.COLORS['LINENO'])
            print(f'\033[1m{self.COLORS["CRITICAL"]}[CRITICAL] {formatted_time} {self.RESET} \033[1m[{formatted_filename}:\033[1m{formatted_lineno}] \033[1m{message}')

