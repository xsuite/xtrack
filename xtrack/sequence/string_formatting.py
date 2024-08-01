import re

UNDERLINE = '\033[4m'
NO_UNDERLINE = '\033[24m'
ERROR = '\033[91m'
WARNING = '\033[93m'
DEFAULT = '\033[39m'


def error(string: str) -> str:
    return ERROR + string + DEFAULT


def warning(string: str) -> str:
    return WARNING + string + DEFAULT


def indent_string(string, indent='    '):
    return re.sub('^', indent, string, flags=re.MULTILINE)