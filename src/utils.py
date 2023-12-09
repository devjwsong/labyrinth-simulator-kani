from inputimeout import inputimeout
from kani.models import ChatMessage, ChatRole
from typing import Any, List, Union

import logging
import string

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


def print_manager_log(msg: str, after_break: bool=False):
    print(f"[Game Manager] Goblin King: {msg}")
    if after_break:
        log_break()


def get_player_input(name: str=None, per_player_time: int=None, after_break: bool=False):
    if name is None:
        query = input("INPUT: ")
    else:
        query = inputimeout(f"[PLAYER] {name.replace('-', ' ')}: ", timeout=per_player_time)
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
def check_init_types(manager):
    # The scene summary.
    try:
        assert isinstance(manager.scene_summary, list), "THE SCENE SUMMARY IS NOT THE LIST TYPE."
        assert len(manager.scene_summary) > 0, "THE SCENE SUMMARY MUST NOT BE EMPTY."

        # The NPCs.
        assert isinstance(manager.npcs, dict), "THE NPCS ATTRIBUTE IS NOT THE DICT TYPE."
        if len(manager.npcs) > 0:
            for name, info in manager.npcs.items():
                assert isinstance(name, str), "THE NAME OF AN NPC IS NOT THE STRING TYPE."
                assert isinstance(info, dict), "THE NPC INFORMATION IS NOT THE DICT TYPE."
                assert isinstance(info['kin'], str), "THE KIN OF AN NPC IS NOT THE STRING TYPE."
                assert isinstance(info['persona'], list), "THE PERSONA OF AN NPC IS NOT THE LIST TYPE."
                assert isinstance(info['goal'], str), "THE GOAL OF AN NPC IS NOT THE STRING TYPE."
                assert isinstance(info['trait'], str), "THE TRAITS OF AN NPC IS NOT THE STRING TYPE."
                assert isinstance(info['flaw'], str), "THE FLAWS OF AN NPC IS NOT THE STRING TYPE."

        # The generation rules.
        assert isinstance(manager.generation_rules, list), "THE LIST OF GENERATION RULES IS NOT THE LIST TYPE."

        # The success condition.
        assert isinstance(manager.success_condition, str), "THE SUCCESS CONDITION IS NOT THE STRING TYPE."
        assert len(manager.success_condition) > 0, "THE SUCCESS CONDITION MUST NOT BE EMPTY."

        # The failure condition.
        assert isinstance(manager.failure_condition, str), "THE FAILURE CONDITION IS NOT THE STRING TYPE."

        # The game flow rules.
        assert isinstance(manager.game_flow, list), "THE LIST OF GAME FLOW RULES IS NOT THE LIST TYPE."

        # The environment.
        assert isinstance(manager.environment, dict), "THE LIST OF ENVIRONMENT SPECIFICATIONS IS NOT THE DICT TYPE."
        if len(manager.environment) > 0:
            for name, desc in manager.environment.items():
                assert isinstance(name, str), "THE NAME OF AN OBJECT IS NOT THE STRING TYPE."
                assert isinstance(desc, str), "THE OBJECT DESCRIPTION IS NOT THE STRING TYPE."
    except AssertionError as e:
        log.error(f"{e}: ASSERTION ERROR.")
        raise Exception()


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
        name = "[Game Manager] Goblin-King"
    if message.role == ChatRole.USER:
        name = f"[Player] {name}"
    if message.role == ChatRole.FUNCTION:
        name = f"[Function] {name}"
    if message.role == ChatRole.SYSTEM:
        if name is None:
            name = "[SYSTEM]"
        else:
            name = f"[SYSTEM] {name}"
    
    return f"{name}: {content}"
