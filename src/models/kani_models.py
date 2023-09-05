from kani.engines.openai import OpenAIEngine

import enum


# Fetching the proper kani engine for the specified model.
def generate_engine(engine_name:str, model_index: str):
    assert engine_name in ['openai', 'huggingface', 'llama', 'vicuna', 'ctransformers', 'llamactransformers'], "Specify a correct engine class name."
    if engine_name == 'openai':
        api_key = input("Enter the API key for OpenAI API: ")
        engine = OpenAIEngine(api_key, model=model_index)
        
    return engine


# Action category.
class Action(enum.Enum):
    # Attacking the opponent.
    ATTACK = 'attack'

    # Dodging the attack.
    DODGE = 'dodge'

    # Defending the attack.
    DEFENSE = 'defense'

    # Using the attacking item.
    ATTACK_ITEM = 'attack_item'

    # Using the healing item.
    HEALING_ITEM = 'healing_item'
