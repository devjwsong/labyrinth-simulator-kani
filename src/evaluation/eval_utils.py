from inputimeout import inputimeout
from kani.models import ChatMessage, ChatRole
from typing import Any, List

import logging
import string
import random
import re

log = logging.getLogger("kani")
message_log = logging.getLogger("kani.messages")


# Trivial definitions for CLI printing.
def print_logic_start(title: str):
    print('#' * 100)
    print(f"{title.upper()}")


def print_question_start():
    print('-' * 100)


def print_system_log(msg: str, after_break: bool=False):
    print(f"[SYSTEM] {msg}")
    if after_break:
        log_break()

def get_player_input(name: str=None, per_player_time: int=None, after_break: bool=False):
    if name is None:
        query = input("INPUT: ")
    else:
        query = inputimeout(f"[PLAYER] {name.replace('_', ' ')}: ", timeout=per_player_time)

    if after_break:
        log_break()
        
    return query


def logic_break():
    print('\n')


def log_break():
    print()


# Default function for allowing a multi-choice query.
def select_options(options: List[Any]):
    while True:
        for o, option in enumerate(options):
            print(f"({o+1}) {option}")
        res = get_player_input(after_break=True)

        try:
            res = int(res)
            if res < 1 or res > len(options):
                print_system_log(f"The allowed value should be from {1} tO {len(options)}.", after_break=True)
            else:
                return res-1
        except ValueError:
            print_system_log("The input should be an integer.", after_break=True)


# Default function for allowing a continuous score value.
def give_score(max_score: float, min_score: float):
    while True:
        print_system_log(f"Give the score between {min_score} - {max_score}.")
        score = get_player_input(after_break=True)

        try:
            score = float(score)
            if score < min_score or score > max_score:
                print_system_log(f"The allowed value should be from {min_score} to {max_score}.", after_break=True)
            else:
                return score
        except ValueError:
            print_system_log("The input should be a valid number.", after_break=True)
