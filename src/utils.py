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

def print_player_log(msg: str, name:str, after_break: bool=False):
    print(f"[PLAYER] {name.replace('_', ' ')}: {msg}")
    if after_break:
        log_break()


def print_manager_log(msg: str, after_break: bool=False):
    print(f"[GAME MANAGER] Goblin King: {msg}")
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
                print_system_log(f"THE ALLOWED VALUE IS FROM {1} TO {len(options)}.", after_break=True)
            else:
                return res-1
        except ValueError:
            print_system_log("THE INPUT SHOULD BE AN INTEGER.", after_break=True)
            

# Function for a randomized/automated multi-choice query. (For simulation)
def select_random_options(options: List[Any]):
    idxs = list(range(len(options)))
    selected = random.choice(idxs)
    return selected


# Removing unnecessary punctuations from the object name.
def remove_punctuation(word: str):
    puncs = list(string.punctuation)
    cut_idx = len(word)
    for i in range(len(word)-1, -1, -1):
        if word[i] in puncs:
            cut_idx = i
        else:
            break
    word = word[:cut_idx]
    return word


# Checking the types of attributes for initialization.
def check_init_types(scene: dict):
    assert isinstance(scene['scene_summary'], list), "THE SCENE SUMMARY IS NOT THE LIST TYPE."
    assert len(scene['scene_summary']) > 0, "THE SCENE SUMMARY MUST NOT BE EMPTY."

    # The NPCs.
    assert isinstance(scene['npcs'], dict), "THE NPCS ATTRIBUTE IS NOT THE DICT TYPE."
    if len(scene['npcs']) > 0:
        for name, info in scene['npcs'].items():
            assert isinstance(name, str), "THE NAME OF AN NPC IS NOT THE STRING TYPE."
            assert isinstance(info, dict), "THE NPC INFORMATION IS NOT THE DICT TYPE."
            assert isinstance(info['kin'], str), "THE KIN OF AN NPC IS NOT THE STRING TYPE."
            assert isinstance(info['persona'], list), "THE PERSONA OF AN NPC IS NOT THE LIST TYPE."
            assert isinstance(info['goal'], str), "THE GOAL OF AN NPC IS NOT THE STRING TYPE."
            assert isinstance(info['trait'], str), "THE TRAITS OF AN NPC IS NOT THE STRING TYPE."
            assert isinstance(info['flaw'], str), "THE FLAWS OF AN NPC IS NOT THE STRING TYPE."

    # The generation rules.
    assert isinstance(scene['generation_rules'], list), "THE LIST OF GENERATION RULES IS NOT THE LIST TYPE."

    # The success condition.
    assert isinstance(scene['success_condition'], str), "THE SUCCESS CONDITION IS NOT THE STRING TYPE."

    # The failure condition.
    assert isinstance(scene['failure_condition'], str), "THE FAILURE CONDITION IS NOT THE STRING TYPE."

    # The game flow rules.
    assert isinstance(scene['game_flow'], list), "THE LIST OF GAME FLOW RULES IS NOT THE LIST TYPE."

    # The environment.
    assert isinstance(scene['environment'], dict), "THE LIST OF ENVIRONMENT SPECIFICATIONS IS NOT THE DICT TYPE."
    if len(scene['environment']) > 0:
        for name, desc in scene['environment'].items():
            assert isinstance(name, str), "THE NAME OF AN OBJECT IS NOT THE STRING TYPE."
            assert isinstance(desc, str), "THE OBJECT DESCRIPTION IS NOT THE STRING TYPE."


# Finding the valid current queries.
def find_current_point(chat_history: List[ChatMessage]) -> int:
    if chat_history[-1].role == ChatRole.USER:
        target_roles = [ChatRole.USER]
    elif chat_history[-1].role == ChatRole.FUNCTION:
        target_roles = [ChatRole.FUNCTION]

    idx = len(chat_history)
    for i in range(len(chat_history)-1, -1, -1):
        if chat_history[i].role in target_roles:
            idx = i
        else:
            break

    return idx


# Converting ChatMessage into a natural language message.
def convert_into_natural(message: ChatMessage):
    name, content = message.name, message.content
    if message.role == ChatRole.ASSISTANT:
        name = f"[Game Manager] {name.replace('_', ' ')}"
    if message.role == ChatRole.USER:
        name = f"[Player] {name.replace('_', ' ')}"
    if message.role == ChatRole.FUNCTION:
        name = f"[Function] {name}"
    if message.role == ChatRole.SYSTEM:
        if name is None:
            name = "[SYSTEM]"
        else:
            name = f"[SYSTEM] {name}"
    
    return f"{name}: {content}"


# Extracting the class index in the output of a classification problem.
def convert_into_class_idx(res: str, options: list):
    pattern = r'\d+'
    matches = re.findall(pattern, res)
    if matches:
        index = int(matches[0])
        if index >= len(options):
            return select_random_options(options)
        return index
    else:
        return select_random_options(options)
