from inputimeout import inputimeout
from typing import Any, List
from eval_constants import RULE_SUMMARY

import logging
import json

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


# A sub-logic for showing the scene state.
def show_scene_state(scene: dict):
    keys = list(scene.keys())
    k = 0
    while k < len(keys):
        options = []
        if k == 0:
            options = ["Next", "Go back to the evaluation"]
        elif k == len(keys)-1:
            options = ["Prev", "Go back to the evaluation"]
        else:
            options = ["Next", "Prev", "Go back to the evaluation"]
        
        key = keys[k]
        print_question_start()
        print_system_log(f"{key}: ")
        print(json.dumps(scene[key], indent=4))
        log_break()

        idx = select_options(options)
        if options[idx] == "Next":
            k += 1
        elif options[idx] == "Prev":
            k -= 1
        else:
            break

# A sub-logic for showing the player states.
def show_player_states(players: list[dict]):
    num_players = len(players)
    p = 0
    while p < num_players:
        if p == 0:
            options = ["Next", "Go back to the evaluation"]
        elif p == num_players-1:
            options = ["Prev", "Go back to the evaluation"]
        else:
            options = ["Next", "Prev", "Go back to the evaluation"]

        player = players[p]
        print_question_start()
        print_system_log(f"Player {p+1}: {player['name']}")
        print(json.dumps(player, indent=4))
        log_break()

        idx = select_options(options)
        if options[idx] == "Next":
            p += 1
        elif options[idx] == "Prev":
            p -= 1
        else:
            break

# A sub-logic for showing the past chat history.
def show_past_history(past_history: list[dict]):
    num_messages = len(past_history)
    m = 0
    while m < num_messages:
        if m == 0:
            options = ["Next", "Go back to the evaluation"]
        elif m == num_messages-1:
            options = ["Prev", "Go back to the evaluation"]
        else:
            options = ["Next", "Prev", "Go back to the evaluation"]
        
        message = past_history[m]
        print_question_start()
        print(json.dumps(message, indent=4))
        log_break()

        idx = select_options(options)
        if options[idx] == "Next":
            m += 1
        elif options[idx] == "Prev":
            m -= 1
        else:
            break


# A sub-logic for showing the current queries.
def show_current_queries(current_queries: list[dict]):
    num_messages = len(current_queries)
    m = 0
    while m < num_messages:
        if m == 0:
            options = ["Next", "Go back to the evaluation"]
        elif m == num_messages-1:
            options = ["Prev", "Go back to the evaluation"]
        else:
            options = ["Next", "Prev", "Go back to the evaluation"]
        
        message = current_queries[m]
        print_question_start()
        print(json.dumps(message, indent=4))
        log_break()

        idx = select_options(options)
        if options[idx] == "Next":
            m += 1
        elif options[idx] == "Prev":
            m -= 1
        else:
            break


# A sub-logic for showing the game rules.
def show_game_rules():
    num_parts = len(RULE_SUMMARY)
    r = 0
    while r < num_parts:
        if r == 0:
            options = ["Next", "Go back to the evaluation"]
        elif r == num_parts-1:
            options = ["Prev", "Go back to the evaluation"]
        else:
            options = ["Next", "Prev", "Go back to the evaluation"]

        part = RULE_SUMMARY[r]
        print_question_start()
        print_system_log(f"Labyrinth's rule ({r+1})")
        print('\n'.join(part))
        log_break()

        idx = select_options(options)
        if options[idx] == "Next":
            r += 1
        elif options[idx] == "Prev":
            r -= 1
        else:
            break
