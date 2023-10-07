from ast import Assert
from kani import Kani
from kani.models import ChatMessage
from typing import Any, List, Dict

import json
import logging

log = logging.getLogger("kani")
message_log = logging.getLogger("kani.messages")


# The whole game manager class.
class GameManager(Kani):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Attributes which should be initialized before the game.
        self.scene_summary = []
        self.npcs = {}
        self.generation_rules = []
        self.success_condition = ""
        self.failure_condition = ""
        self.game_flow = []
        self.environment = []
        self.random_tables = {}
        self.consequences = ""

    # Initialization of the scene.
    async def init_scene(self, init_query: str, scene: Dict[str, Any], **kwargs):
        query = f"{init_query}\n{scene}"
        res = await self.chat_round_str(query, **kwargs)

        # Finding each key and mapping into the corresponding attribute.
        try:
            res = json.loads(res)

            self.scene_summary = res['scene_summary']
            self.npcs = res['npcs']
            self.generation_rules = res['generation_rules']
            self.success_condition = res['success_condition']
            self.failure_condition = res['failure_condition']
            self.game_flow = res['game_flow']
            self.environment = res['environment']
            self.random_tables = scene['random_tables']
            self.consequences = scene['consequences']

            self.check_types()

        except json.decoder.JSONDecodeError as e:
            log.debug(res)
            log.error(f"{e}: The output format cannot be converted into dict.")
            raise Exception()
            # TODO: Fixing from the model if there is a JSON parsing error.
        except KeyError as e:
            log.debug(res)
            log.error(f"{e}: Missing key.")
            raise Exception()
        
        # Initialization record should be removed.
        self.chat_history = []

    # Checking the types of attributes for initialization.
    def check_types(self):
        # The scene summary.
        try:
            assert isinstance(self.scene_summary, list), "The scene summary is not the list type."
            assert len(self.scene_summary) > 0, "The scene summary must not be empty."

            # The NPCs.
            assert isinstance(self.npcs, dict), "The npcs attribute is not the dict type."
            if len(self.npcs) > 0:
                for name, info in self.npcs.items():
                    assert isinstance(name, str), "The name of an NPC is not the string type."
                    assert isinstance(info, dict), "The NPC information is not the dict type."
                    assert isinstance(info['kin'], str), "The kin of an NPC is not the string type."
                    assert isinstance(info['persona'], list), "The persona of an NPC is not the list type."
                    assert isinstance(info['goal'], str), "The goal of an NPC is not the string type."
                    assert isinstance(info['trait'], str), "The traits of an NPC is not the string type."
                    assert isinstance(info['flaw'], str), "The flaws of an NPC is not the string type."

            # The generation rules.
            assert isinstance(self.generation_rules, list), "The list of generation rules is not the list type."

            # The success condition.
            assert isinstance(self.success_condition, str), "The success condition is not the string type."
            assert len(self.success_condition) > 0, "The success condition must not be empty."

            # The failure condition.
            assert isinstance(self.failure_condition, str), "The failure condition is not the string type."

            # The game flow rules.
            assert isinstance(self.game_flow_rules, list), "The list of game flow rules is not the list type."

            # The environment.
            assert isinstance(self.environment, list), "The list of environment specifications is not the list type."
        except AssertionError as e:
            log.error(f"{e}: Assertion error.")
            raise Exception()

    # Showing the scene information which the manager has initialized.
    def show_scene(self):
        print("<SCENE SUMMARY>")
        print(self.scene_summary)

        print("<NPCS>")
        print(self.npcs)

        print("<GENERATION RULES>")
        print(self.generation_rules)

        print("<SUCCESS CONDITION>")
        print(self.success_condition)

        print("<FAILURE CONDITION>")
        print(self.failure_condition)

        print("<GAME FLOW>")
        print(self.game_flow)

        print("<ENVIRONMENT>")
        print(self.environment)

        print("<RANDOM TABLES>")
        print(self.random_tables)

        print("<CONSEQUENCES>")
        print(self.consequences)

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
