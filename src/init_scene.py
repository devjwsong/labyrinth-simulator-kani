from utils import print_question_start, print_system_log, get_player_input, log_break
from constants import (
    SCENE_INIT_PROMPT,
    RULE_SUMMARY,
    SCENE_SUMMARY_DETAILS,
    NPC_DETAILS,
    SUCCESS_CONDITION_DETAILS,
    FAILURE_CONDITION_DETAILS,
    GAME_FLOW_DETAILS,
    ENVIRONMENT_DETAILS,
    RANDOM_TABLE_DETAILS
)
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
import ast
import time

log = logging.getLogger("kani")
message_log = logging.getLogger("kani.messages")
WAITING_TIME = 30  # Waiting time for avoiding the rate limit exceeding.


# Checking the types of attributes for initialization.
def check_init_types(scene: dict):
    assert isinstance(scene['scene_summary'], list), "THE SCENE SUMMARY IS NOT THE LIST TYPE."
    assert len(scene['scene_summary']) > 0, "THE SCENE SUMMARY MUST NOT BE EMPTY."

    # The NPCs.
    assert isinstance(scene['npcs'], dict), "THE NPCS ATTRIBUTE IS NOT THE DICT TYPE."
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
    for name, desc in scene['environment'].items():
        assert isinstance(name, str), "THE NAME OF AN OBJECT IS NOT THE STRING TYPE."
        assert isinstance(desc, str), "THE OBJECT DESCRIPTION IS NOT THE STRING TYPE."

    # The random tables.
    assert isinstance(scene['random_tables'], dict), "THE LIST OF RANDOM TABLES IS NOT THE DICT TYPE."
    for name, table in scene['random_tables'].items():
        assert isinstance(name, str), "THE NAME OF A TABLE IS NOT THE STRING TYPE."
        assert isinstance(table, list), "THE TABLE IS NOT THE LIST TYPE."
        for entry in table:
            assert isinstance(entry, str), "AN ENTRY IN A TABLE IS NOT STRING TYPE."


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
    try:
        result = {}
        generation_params = {
            'temperature': 0,
            'top_p': 1,
            'presence_penalty': 0,
        }

        # Generating the scene summary.
        res = await agent.chat_round_str(f"{' '.join(SCENE_SUMMARY_DETAILS)}\nScene: {scene}", **generation_params)
        print(res)
        result['scene_summary'] = ast.literal_eval(res)
        time.sleep(WAITING_TIME)

        # Generating the NPCs.
        res = await agent.chat_round_str(f"{' '.join(NPC_DETAILS)}\nScene: {scene}", response_format={'type': 'json_object'}, **generation_params)
        print(res)
        result['npcs'] = json.loads(res)
        time.sleep(WAITING_TIME)
        
        # Generating the success condition.
        res = await agent.chat_round_str(f"{' '.join(SUCCESS_CONDITION_DETAILS)}\nScene: {scene}")
        print(res)
        result['success_condition'] = res
        time.sleep(WAITING_TIME)

        # Generating the failure condition.
        res = await agent.chat_round_str(f"{' '.join(FAILURE_CONDITION_DETAILS)}\nScene: {scene}")
        print(res)
        result['failure_condition'] = res
        time.sleep(WAITING_TIME)

        # Generating the game flow.
        res = await agent.chat_round_str(f"{' '.join(GAME_FLOW_DETAILS)}\nScene: {scene}", **generation_params)
        print(res)
        result['game_flow'] = ast.literal_eval(res)
        time.sleep(WAITING_TIME)

        # Generating the environment.
        res = await agent.chat_round_str(f"{' '.join(ENVIRONMENT_DETAILS)}\nScene: {scene}", response_format={'type': 'json_object'}, **generation_params)
        print(res)
        result['environment'] = json.loads(res)
        time.sleep(WAITING_TIME)

        # Generating the random tables.
        res = await agent.chat_round_str(f"{' '.join(RANDOM_TABLE_DETAILS)}\nScene: {scene}", response_format={'type': 'json_object'}, **generation_params)
        print(res)
        result['random_tables'] = json.loads(res)

        result['chapter'] = scene['chapter']
        result['scene'] = scene['scene']
        result['consequences'] = scene['consequences']

        # Checking the data types generated.
        check_init_types(result)

        await agent.engine.close()

        return result

    except json.decoder.JSONDecodeError as e:  # JSON parsing error: This should be noted as 0 for the evaluation.
        log.error(f"{e}: The output format cannot be converted into dict.")
        raise Exception()
    except KeyError as e:  # Missing attributes error: This should be noted as 0.2 for the evaluation.
        log.error(f"{e}: Missing key.")
        raise Exception()
    except AssertionError as e:  # Incorrect types error: This should be noted as 0.5 for the evaluation.
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

    system_prompt = ' '.join(SCENE_INIT_PROMPT)
    rule_content = '\n'.join([' '.join(part) for part in RULE_SUMMARY])  # For scene initialization, we only use the direct injection of rules.
    rule_prompt = ChatMessage.system(name="Game_Rules", content=rule_content)
    agent = Kani(engine, chat_history=[rule_prompt], system_prompt=system_prompt)

    # Running the main logic.
    result = asyncio.run(init_scene(args, agent))

    # Exporting the result.
    export_result(result, args.model_idx, args.scene_idx, username, execution_time)
