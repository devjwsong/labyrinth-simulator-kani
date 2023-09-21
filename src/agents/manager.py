from kani import Kani, AIParam
from kani.models import ChatMessage
from typing import Annotated, Any, List, Dict

import json
import logging

log = logging.getLogger("kani")
message_log = logging.getLogger("kani.messages")


# The whole game manager class.
class GameManager(Kani):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Attributes which should be initialized before the game.
        self.setting = []
        self.npcs = {}
        self.generation_rules = []
        self.success_condition = ""
        self.failure_condition = ""
        self.random_tables = {}

    # Initialization of the scene.
    async def init_scene(self, init_query: str, scene: Dict[str, Any]):
        query = f"{init_query}\n{scene}"
        res = await self.chat_round_str(query)

        # Finding each key and mapping into the corresponding attribute.
        try:
            res = json.loads(res)

            self.setting = res['setting']
            self.npcs = res['npcs']
            self.generation_rules = res['generation_rules']
            self.success_condition = res['success_condition']
            self.failure_condition = res['failure_condition']
            self.random_tables = res['random_tables']

            self.check_types()

        except json.decoder.JSONDecodeError as e:
            log.debug(res)
            log.error(f"{e}: The output format cannot be converted into dict.")
        except KeyError as e:
            log.debug(res)
            log.error(f"{e}: Missing key.")

    # Checking the types of attributes for initialization.
    def check_types(self):
        # The scene description.
        assert isinstance(self.setting, list), "The scene description is not the list type."

        # The NPCs.
        assert isinstance(self.npcs, dict), "The npc dictionary is not the dict type."
        if len(self.npcs) > 0:
            for name, info in self.npcs.items():
                assert isinstance(name, str), "The name of an NPC is not the string type."
                assert isinstance(info, dict), "The NPC information is not the dict type."
                assert isinstance(info['persona'], list), "The persona of an NPC is not the list type."
                assert isinstance(info['goal'], str), "The goal of an NPC is not the string type."

        # The generation rules.
        assert isinstance(self.generation_rules, list), "The list of generation rules is not the list type."

        # The success condition.
        assert isinstance(self.success_condition, str), "The success condition is not the string type."

        # The failure condition.
        assert isinstance(self.failure_condition, str), "The failure condition is not the string type."

        # The random tables.
        assert isinstance(self.random_tables, dict), "The random table dictionary is not the dict type."
        if len(self.random_tables) > 0:
            for name, table in self.random_tables.items():
                assert isinstance(name, str), "The name of a table is not the string type."
                assert isinstance(table, list), "The table is not the list type."

    # Converting the generation result into the binary answer.
    def translate_into_binary(self, response: str):
        if 'yes' in response.lower():
            return True
        elif 'no' in response.lower():
            return False
        else:
            return None

    # Validating the generated response from the NPC.
    async def validate_generation_rule(self, chat_history: List[ChatMessage]):
        # The default system prompt consists of the instruction and the predefined generation rules.
        system_prompt = "You are the game manager in a fantasy text adventure game. " + \
            "You should determine whether the last response from the NPC follows the defined rule. " + \
            "You are given the dialogue history so far between the player(user) and the NPC(assistant) for reference. " + \
            "You must answer only either 'yes' or 'no'. "
        system_prompt += "Rules: "
        for r, rule in enumerate(self.generation_rules):
            system_prompt += f"{r+1} - {rule} "
        system_prompt = system_prompt[:-1]

        kani = Kani(self.engine, chat_history=chat_history, system_prompt=system_prompt)
        response = await kani.chat_round_str("Does the last response follow the rules?")

        return self.translate_into_binary(response)

    # Validating if the current interaction falls into the success condition.
    async def validate_success_condition(self, chat_history: List[ChatMessage]):
        # The default system prompt consists of the instruction and the success condition.
        system_prompt = "You are the game manager in a fantasy text adventure game. " + \
            "You should determine whether the current game state satisfies the success condition of the player(user). " + \
            "You are given the dialogue history so far between the player(user) and the NPC(assistant) for reference. " + \
            "You must answer only either 'yes' or 'no'. "
        system_prompt += f"Success condition: {self.success_condition}"

        kani = Kani(self.engine, chat_history=chat_history, system_prompt=system_prompt)
        response = await kani.chat_round_str("Have the player accomplished the success condition?")

        return self.translate_into_binary(response)
    
    # Validating if the current interaction falls into the failure condition.
    async def validate_failure_condition(self, chat_history: List[ChatMessage]):
        # The default system prompt consists of the instruction and the failure condition.
        system_prompt = "You are the game manager in a fantasy text adventure game. " + \
            "You should determine whether the current game state satisfies the failure condition of the player(user). " + \
            "You are given the dialogue history so far between the player(user) and the NPC(assistant) for reference. " + \
            "You must answer only either 'yes' or 'no'. "
        system_prompt += f"Failure condition: {self.failure_condition}"

        kani = Kani(self.engine, chat_history=chat_history, system_prompt=system_prompt)
        response = await kani.chat_round_str("Have the player fallen into the failure condition?")

        return self.translate_into_binary(response)
