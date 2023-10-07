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
        self.chapter = ""
        self.scene = ""
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

            self.chapter = scene['chapter']
            self.scene = scene['scene']
            self.scene_summary = res['scene_summary']
            self.npcs = res['npcs']
            self.generation_rules = res['generation_rules']
            self.success_condition = res['success_condition']
            self.failure_condition = res['failure_condition']
            self.game_flow = res['game_flow']
            self.environment = res['environment']
            self.random_tables = scene['random_tables']
            self.consequences = scene['consequences']

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

    # Showing the scene information which the manager has initialized.
    def show_scene(self):
        print("<CHAPTER>")
        print(self.chapter)

        print("<SCENE>")
        print(self.scene)

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
