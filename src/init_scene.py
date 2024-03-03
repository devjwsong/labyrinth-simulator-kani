from utils import print_question_start, print_system_log, get_player_input, log_break
from constants import SCENE_INIT_PROMPT, RULE_SUMMARY
from kani import Kani
from kani.engines.openai import OpenAIEngine
from kani.models import ChatMessage
from datetime import datetime
from pytz import timezone
from argparse import Namespace

import argparse
import json
import asyncio
import logging
import os

log = logging.getLogger("kani")
message_log = logging.getLogger("kani.messages")


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


# Exporting the initialized result.
def export_result(result: dict, model_idx: str, scene_idx: int, username: str, execution_time: str):
    file_dir = f"scenes/scene={scene_idx}"
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    file_path = f"{file_dir}/{username}-model={model_idx}-time={execution_time}.json"
    with open(file_path, 'w') as f:
        json.dump(result, f)

    print_system_log(f"THE INITIALIZED SCENE {scene_idx} HAS BEEN EXPORTED TO {file_path}.")


# The main logic for scene initialization.
async def init_scene(args: Namespace, agent: Kani):
    with open("data/scenes.json", 'r') as f:
        scenes = json.load(f)
    assert args.scene_idx is not None, "The scene index should be provided."
    assert 0 <= args.scene_idx < len(scenes), "The scene index is not valid."

    scene = scenes[args.scene_idx]

    print_system_log("INITIALIZING THE SCENE...")
    raw_res = await agent.chat_round_str(f"Generate the JSON output for the initialized scene attributes.\nScene: {scene}", 
        response_format={ "type": "json_object" }
    )

    # Finding each key and mapping into the corresponding attribute.
    try:
        result = json.loads(raw_res)

        result['chapter'] = scene['chapter']
        result['scene'] = scene['scene']
        result['random_tables'] = scene['random_tables']
        result['consequences'] = scene['consequences']

        # Checking the data types generated.
        check_init_types(result)

        await engine.client.close()

        return result

    except json.decoder.JSONDecodeError as e:  # JSON parsing error: This should be noted as 0 for the evaluation.
        log.debug(raw_res)
        log.error(f"{e}: The output format cannot be converted into dict.")
        raise Exception()
    except KeyError as e:  # Missing attributes error: This should be noted as 0.2 for the evaluation.
        log.debug(result)
        log.error(f"{e}: Missing key.")
        raise Exception()
    except AssertionError as e:  # Incorrect types error: This should be noted as 0.5 for the evaluation.
        log.debug(result)
        log.error(f"{e}: Incorrect data type.")
        raise Exception()

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_idx', type=str, required=True, help="The index of the model.")
    parser.add_argument('--scene_idx', type=int, required=True, help="The index of the scene to generate.")

    args = parser.parse_args()

    print_question_start()
    print_system_log("GIVE US YOUR USERNAME, WHICH IS USED FOR RECORDING PURPOSE.")
    username = get_player_input(after_break=True)

    now = datetime.now(timezone('US/Eastern'))
    execution_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    # Setting the kani.
    print_question_start()
    api_key = input("Enter the API key for OpenAI API: ")
    log_break()
    engine = OpenAIEngine(api_key, model=args.model_idx)

    system_prompt = '\n'.join([' '. join(part) for part in SCENE_INIT_PROMPT])
    rule_content = '\n'.join([' '.join(part) for part in RULE_SUMMARY])  # For scene initialization, we only use the direct injection of rules.
    rule_prompt = ChatMessage.system(name="Game_Rules", content=rule_content)
    agent = Kani(engine, chat_history=[rule_prompt], system_prompt=system_prompt)

    # Running the main logic.
    result = asyncio.run(init_scene(args, agent))

    # Exporting the result.
    export_result(result, args.model_idx, args.scene_idx, username, execution_time)
