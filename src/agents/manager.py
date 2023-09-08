from kani import Kani
from kani.models import ChatMessage
from typing import List
from src.models.kani_models import Action, generate_engine

# The whole game manager class.
class GameManager(Kani):
    def __init__(self, rules: List[str], success_condition: str, failure_condition: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Properties which are not part of kani system.
        self.rules = rules
        self.success_condition = success_condition
        self.failure_condition = failure_condition

    # Converting the generation result into the binary answer.
    def translate_into_binary(self, response: str):
        print(response)  # debugging.
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
        for r, rule in enumerate(self.rules):
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
