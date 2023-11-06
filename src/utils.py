from inputimeout import inputimeout
from typing import Any, List

import logging

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
    print(f"[GOBLIN KING] {msg}")
    if after_break:
        log_break()


def get_player_input(name: str=None, per_player_time: int=None, after_break: bool=False):
    if name is None:
        query = input("INPUT: ")
    else:
        query = inputimeout(f"[PLAYER / {name.upper()}]: ", timeout=per_player_time)
    if after_break:
        log_break()
    return query


def logic_break():
    print('\n')


def log_break():
    print()


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
