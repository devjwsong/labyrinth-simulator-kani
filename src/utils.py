from agents.manager import GameManager
from typing import Any, List

import logging

log = logging.getLogger("kani")
message_log = logging.getLogger("kani.messages")


def select_options(options: List[Any]):
    while True:
        for o, option in enumerate(options):
            print(f"{o+1}. {option}")
        res = input("Input: ")

        try:
            res = int(res)
            if res < 1 or res > len(options):
                print(f"The allowed value is from {1} to {len(options)}.")
            else:
                return options[res-1]
        except ValueError:
            print("The input should be an integer.")
            

# Checking the types of attributes for initialization.
def check_init_types(manager: GameManager):
    # The scene summary.
    try:
        assert isinstance(manager.scene_summary, list), "The scene summary is not the list type."
        assert len(manager.scene_summary) > 0, "The scene summary must not be empty."

        # The NPCs.
        assert isinstance(manager.npcs, dict), "The npcs attribute is not the dict type."
        if len(manager.npcs) > 0:
            for name, info in manager.npcs.items():
                assert isinstance(name, str), "The name of an NPC is not the string type."
                assert isinstance(info, dict), "The NPC information is not the dict type."
                assert isinstance(info['kin'], str), "The kin of an NPC is not the string type."
                assert isinstance(info['persona'], list), "The persona of an NPC is not the list type."
                assert isinstance(info['goal'], str), "The goal of an NPC is not the string type."
                assert isinstance(info['trait'], str), "The traits of an NPC is not the string type."
                assert isinstance(info['flaw'], str), "The flaws of an NPC is not the string type."

        # The generation rules.
        assert isinstance(manager.generation_rules, list), "The list of generation rules is not the list type."

        # The success condition.
        assert isinstance(manager.success_condition, str), "The success condition is not the string type."
        assert len(manager.success_condition) > 0, "The success condition must not be empty."

        # The failure condition.
        assert isinstance(manager.failure_condition, str), "The failure condition is not the string type."

        # The game flow rules.
        assert isinstance(manager.game_flow, list), "The list of game flow rules is not the list type."

        # The environment.
        assert isinstance(manager.environment, list), "The list of environment specifications is not the list type."
    except AssertionError as e:
        log.error(f"{e}: Assertion error.")
        raise Exception()
