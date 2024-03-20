from utils import print_question_start, print_system_log, get_player_input, log_break, convert_into_class_idx, convert_into_number
from constants import (
    SCENE_INIT_PROMPT,
    SCENE_SUMMARY_DETAILS,
    NPC_DETAILS,
    SUCCESS_CONDITION_DETAILS,
    FAILURE_CONDITION_DETAILS,
    GAME_FLOW_DETAILS,
    ENVIRONMENT_DETAILS,
    RANDOM_TABLES_DETAILS
)
from kani import Kani
from kani.engines.openai import OpenAIEngine
from datetime import datetime
from pytz import timezone
from argparse import Namespace

import argparse
import json
import asyncio
import logging
import os
import ast
import random

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
def export_result(result: dict, seed: int, model_idx: str, scene_idx: int, username: str, execution_time: str):
    file_dir = f"scenes/scene={scene_idx}/model={model_idx}"
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    file_path = f"{file_dir}/{username}-seed={seed}-time={execution_time}.json"
    with open(file_path, 'w') as f:
        json.dump(result, f)

    print_system_log(f"THE INITIALIZED SCENE {scene_idx} HAS BEEN EXPORTED TO {file_path}.")


# The sub logic for processing a random table.
async def process_random_tables(agent: Kani, scene: dict, random_tables: dict, **generation_params):
    result = {
        'npc_ingredients': {},
        'env_ingredients': {}
    }
    init_tables = {}

    if len(random_tables) == 0:
        return result, random_tables

    # 1: Determining the usages of the random tables.
    options = [
        "Used for initializing the NPCs before the game.",
        "Used for initializing the environmental objects before the game.",
        "Used for both initializing NPCs and environmental objects.",
        "Used for in-game notifications during the game. (e.g. hint, puzzle, or outcomes, etc.)"
    ]
    options_str = '\n'.join([f"{o}: {option}" for o, option in enumerate(options)])
    query = ' '.join(RANDOM_TABLES_DETAILS[0])
    res = await agent.chat_round_str(f"{query}\n\nScene input: {scene}\n\nRandom tables: {random_tables}\n\n{options_str}", **generation_params)
    try:
        usages = json.loads(res)
        for k, v in usages.items():
            idx = v
            if not isinstance(idx, int):
                idx = convert_into_class_idx(idx)
            if idx != len(options)-1:
                init_tables[k] = 0

            if idx == 0:
                result['npc_ingredients'][k] = []
            if idx == 1:
                result['env_ingredients'][k] = []
            if idx == 2:
                result['npc_ingredients'][k] = []
                result['env_ingredients'][k] = []

    except json.decoder.JSONDecodeError as e:
        log.debug(res)
        log.error(f"{e}: The output format cannot be converted into dict.")
        raise Exception()

    print(f"Usage result: {usages}")

    # There is no tables for initialization.
    if len(init_tables) == 0:
        return result, random_tables

    # 2. Determining the number of entries to retrieve.
    query = ' '.join(RANDOM_TABLES_DETAILS[1])
    res = await agent.chat_round_str(f"{query}\n\nScene input: {scene}\n\nRandom tables: {list(init_tables.keys())}", **generation_params)
    try:
        dists = json.loads(res)
        for k, v in dists.items():
            num_samples = v
            if not isinstance(num_samples, int):
                num_samples = convert_into_number(num_samples)
            init_tables[k] = num_samples
    except json.decoder.JSONDecodeError as e:
        log.debug(res)
        log.error(f"{e}: The output format cannot be converted into dict.")
        raise Exception()

    print(f"# of samples: {dists}")

    # 3. Sampling the entries.
    for table_name, num_samples in init_tables.items():
        samples = random.sample(random_tables[table_name], num_samples)
        random_tables.pop(table_name)

        if table_name in result['npc_ingredients']:
            result['npc_ingredients'][table_name] = samples
        if table_name in result['env_ingredients']:
            result['env_ingredients'][table_name] = samples

    print(result)

    return result, random_tables


# The main logic for scene initialization.
async def init_scene(args: Namespace, agent: Kani):
    with open("data/scenes.json", 'r') as f:
        scenes = json.load(f)
    assert args.scene_idx is not None, "The scene index should be provided."
    assert 0 <= args.scene_idx < len(scenes), "The scene index is not valid."

    scene = scenes[args.scene_idx]

    print_system_log("INITIALIZING THE SCENE...")
    try:
        result = {
            'chapter': scene['chapter'],
            'scene': scene['scene'],
            'consequences': scene['consequences']
        }
        generation_params = {
            'temperature': 0.2,
            'top_p': 1,
            'presence_penalty': 0,
            'frequency_penalty': 0,
        }

        random_tables = scene['random_tables']
        scene.pop('random_tables')

        # Processing the random tables first.
        processed_tables, random_tables = await process_random_tables(agent, scene, random_tables, **generation_params)
        result['random_tables'] = random_tables
        scene.update(processed_tables)

        # Generating the scene summary.
        res = await agent.chat_round_str(f"{' '.join(SCENE_SUMMARY_DETAILS)}\n\nScene: {scene}", **generation_params)
        print(res)
        result['scene_summary'] = ast.literal_eval(res)

        # Generating the NPCs.
        res = await agent.chat_round_str(f"{' '.join(NPC_DETAILS)}\n\nScene: {scene}", **generation_params)
        print(res)
        result['npcs'] = json.loads(res)
        
        # Generating the success condition.
        res = await agent.chat_round_str(f"{' '.join(SUCCESS_CONDITION_DETAILS)}\n\nScene: {scene}")
        print(res)
        result['success_condition'] = res

        # Generating the failure condition.
        res = await agent.chat_round_str(f"{' '.join(FAILURE_CONDITION_DETAILS)}\n\nScene: {scene}")
        print(res)
        result['failure_condition'] = res

        # Generating the game flow.
        res = await agent.chat_round_str(f"{' '.join(GAME_FLOW_DETAILS)}\n\nScene: {scene}", **generation_params)
        print(res)
        result['game_flow'] = ast.literal_eval(res)

        # Generating the environment.
        res = await agent.chat_round_str(f"{' '.join(ENVIRONMENT_DETAILS)}\n\nScene: {scene}\n\nGenerated NPCs: {result['npcs']}", **generation_params)
        print(res)
        result['environment'] = json.loads(res)

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

    parser.add_argument('--seed', type=int, required=True, help="The random seed.")
    parser.add_argument('--model_idx', type=str, required=True, help="The index of the model.")
    parser.add_argument('--scene_idx', type=int, required=True, help="The index of the scene to generate.")

    args = parser.parse_args()

    print_question_start()
    print_system_log("GIVE US YOUR USERNAME, WHICH IS USED FOR RECORDING PURPOSE.")
    username = get_player_input(after_break=True)

    now = datetime.now(timezone('US/Eastern'))
    execution_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    random.seed(args.seed)

    # Setting the kani.
    print_question_start()
    api_key = input("Enter the API key for OpenAI API: ")
    log_break()
    engine = OpenAIEngine(api_key, model=args.model_idx)

    system_prompt = ' '.join(SCENE_INIT_PROMPT)
    agent = Kani(engine, system_prompt=system_prompt)

    # Running the main logic.
    result = asyncio.run(init_scene(args, agent))

    # Exporting the result.
    export_result(result, args.seed, args.model_idx, args.scene_idx, username, execution_time)
