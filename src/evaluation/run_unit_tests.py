import sys
sys.path.insert(0, "/home/devjwsong/labyrinth-simulator-kani/src")

from kani.engines.openai import OpenAIEngine
from kani.utils.message_formatters import assistant_message_contents_thinking
from argparse import Namespace
from datetime import datetime
from copy import deepcopy
from pytz import timezone
from eval_utils import print_system_log, print_manager_log, log_break, convert_into_message, convert_into_dict, convert_into_natural
from eval_constants import ASSISTANT_INSTRUCTION
from agents.manager import GameManager
from agents.player import Player

import argparse
import os
import json
import asyncio
import torch
import time


# Comparing between the prediction and answer.
def get_score(updated: dict, pred_states: dict, output_states: dict):
    TP = 0
    score_per_update = 100
    for obj in updated:
        name = obj['function']

        if name == 'activate_action_scene' or name == 'terminate_action_scene':
            if pred_states['scene']['is_action_scene'] == output_states['scene']['is_action_scene']:
                TP += score_per_update

        if name == 'create_npc':
            if len(pred_states['scene']['npcs']) == len(output_states['scene']['npcs']):
                TP += (score_per_update / 2)

                is_type_correct = True
                for k, v in pred_states['scene']['npcs'].items():
                    if not isinstance(v, dict) or not isinstance(k, str):
                        is_type_correct = False
                        break

                    if 'kin' not in v or not isinstance(v['kin'], str): 
                        is_type_correct = False
                        break

                    if 'persona' not in v or not isinstance(v['persona'], list):
                        is_type_correct = False
                        break

                    if 'goal' not in v or not isinstance(v['goal'], str):
                        is_type_correct = False
                        break

                    if 'trait' not in v or not isinstance(v['trait'], str):
                        is_type_correct = False
                        break

                    if 'flaw' not in v or not isinstance(v['flaw'], str):
                        is_type_correct = False
                        break
                
                if is_type_correct:
                    TP += (score_per_update / 2)

        if (name == 'add_trait' or name == 'add_flaw' or name == 'add_item' or 
                name == 'remove_trait' or name == 'remove_flaw' or name == 'remove_item' or 
                name == 'use_item' or name == 'use_environment'):
            player_name = obj['arguments']['player_name']
            player_idx = -1
            for p, player in enumerate(output_states['players']):
                if player['name'] == player_name:
                    player_idx = p
                    break
            
            if name == 'add_trait':
                if len(pred_states['players'][player_idx]['traits']) == len(output_states['players'][player_idx]['traits']):
                    TP += score_per_update

            if name == 'add_flaw':
                if len(pred_states['players'][player_idx]['flaws']) == len(output_states['players'][player_idx]['flaws']):
                    TP += score_per_update

            if name == 'add_item':
                if len(pred_states['players'][player_idx]['inventory']) == len(output_states['players'][player_idx]['inventory']):
                    TP += score_per_update

            if name == 'remove_trait':
                if pred_states['players'][player_idx]['traits'] == output_states['players'][player_idx]['traits']:
                    TP += score_per_update

            if name == 'remove_flaw':
                if pred_states['players'][player_idx]['flaws'] == output_states['players'][player_idx]['flaws']:
                    TP += score_per_update

            if name == 'remove_item' or name == 'use_environment':
                if (pred_states['players'][player_idx]['inventory'] == output_states['players'][player_idx]['inventory'] and 
                        pred_states['scene']['environment'] == output_states['scene']['environment']):
                    TP += score_per_update

            if name == 'use_item':
                if pred_states['players'][player_idx]['inventory'] == output_states['players'][player_idx]['inventory']:
                    TP += score_per_update

        if name == 'add_object':
            if len(pred_states['scene']['environment']) == len(output_states['scene']['environment']):
                TP += score_per_update

        if name == 'use_random_table':
            if pred_states['scene']['random_tables'].keys() == output_states['scene']['random_tables'].keys():
                TP += (score_per_update / 2) 

                is_size_same = True
                for k, v in output_states['scene']['random_tables'].items():
                    if len(v) != len(pred_states['scene']['random_tables'][k]):
                        is_size_same = False
                        break

                if is_size_same:
                    TP += (score_per_update / 2) 

    return TP / len(updated)

# Main logic for a unit test.
async def test(args: Namespace, engine: OpenAIEngine, unit_test: dict):
    input_states, output_states, dialogue, updated = deepcopy(unit_test['input']), deepcopy(unit_test['output']), deepcopy(unit_test['dialogue']), deepcopy(unit_test['updated'])

    # Setting the game manager and scene.
    system_prompt = ' '.join(ASSISTANT_INSTRUCTION)
    manager = GameManager(
        scene=input_states['scene'],
        main_args=args,
        engine=engine, 
        system_prompt=system_prompt
    )

    # Setting the player AIs.
    players = []
    for data in input_states['players']:
        player = Player(
            name=data['name'],
            kin=data['kin'],
            persona=data['persona'],
            goal=data['goal'],
            traits=data['traits'],
            flaws=data['flaws'],
            inventory=data['inventory'],
            additional_notes=data['additional_notes']
        )
        players.append(player)

    manager.players = players
    manager.name_to_idx = {player.name: idx for idx, player in enumerate(players)}

    for message in dialogue:
        print(convert_into_natural(message))

    # Let the states updated.
    gen_count = 0
    async for response in manager.full_round(
        [convert_into_message(message) for message in dialogue],
        max_tokens=args.max_tokens,
        include_functions=args.include_functions,
        include_rules=args.include_rules,
        include_scene_state=args.include_scene_state,
        include_player_states=args.include_player_states,
        generate_states=args.generate_states,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        temperature=args.temperature,
        top_p=args.top_p
    ):        
        print(convert_into_natural(convert_into_dict(response)))

        gen_count += 1
        if gen_count == 10:
            break
    
    updated_dialogue = deepcopy(manager.current_queries)
    pred_states = manager.make_context()
    res = get_score(updated, pred_states, output_states)

    # Freeing memory.
    del manager.encoder
    torch.cuda.empty_cache()

    return {
        'score': res,
        'input': unit_test['input'],
        'output': unit_test['output'],
        'predicted': pred_states,
        'dialogue': [convert_into_dict(message) for message in updated_dialogue],
        'updated': updated
    }

if __name__=='__main__':
    now = datetime.now(timezone('US/Eastern'))
    execution_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    parser = argparse.ArgumentParser()

    # Arguments for the gameplay.
    parser.add_argument('--model_idx', type=str, required=True, help="The index of the model.")
    parser.add_argument('--rule_injection', type=str, default='full', help="The rule injection policy.")
    parser.add_argument('--tests_path', type=str, required=True, help="The path of the JSON file which has the unit tests.")
    parser.add_argument('--result_dir', type=str, default="unit_test_results", help="The directory of the exported test results.")

    # Parameters for the prompt construction.
    parser.add_argument('--concat_policy', type=str, default='simple', help="The concatenation policy for including the previous chat logs.")
    parser.add_argument('--max_num_msgs', type=int, help="The maximum number of messages to be included in the prompt as chat history.")
    parser.add_argument('--summarization', action='store_true', help="Setting whether to include the summarization or not.")
    parser.add_argument('--summ_period', type=int, help="The summarization period in terms of the number of turns.")
    parser.add_argument('--clear_raw_logs', action='store_true', help="Setting whether to remove the raw chat logs after the summarization.")

    # Parameters for toggling the additional contexts.
    parser.add_argument('--include_functions', action='store_true', help="Setting whether to use function calls or not.")
    parser.add_argument('--include_rules', action='store_true', help="Setting whether to include the game rules in the prompt.")
    parser.add_argument('--include_scene_state', action='store_true', help="Setting whether to include the state of the current scene.")
    parser.add_argument('--include_player_states', action='store_true', help="Setting whether to include the states of the players.")
    parser.add_argument('--generate_states', action='store_true', help="Setting whether to use a model to directly generate the scene/player states.")

    # Parameters for the response generation.
    parser.add_argument('--max_tokens', type=int, help="The maximum number of tokens to generate.")
    parser.add_argument('--frequency_penalty', type=float, default=0.5, help="A positive value penalizes the repetitive new tokens. (-2.0 - 2.0)")
    parser.add_argument('--presence_penalty', type=float, default=0.5, help="A positive value penalizes the new tokens based on whether they appear in the text so far. (-2.0 - 2.0)")
    parser.add_argument('--temperature', type=float, default=0.5, help="A higher value makes the output more random. (0.0 - 2.0)")
    parser.add_argument('--top_p', type=float, default=1.0, help="The probability mass which will be considered for the nucleus sampling. (0.0 - 1.0)")

    args = parser.parse_args()

    assert args.rule_injection in ['full', 'retrieval'], "Specify an available rule injection option: 'full' / 'retrieval', or leave it as non-specified."
    if not args.summarization:
        assert args.summ_period is None, "To use summ_period, you must set the summarization argument."
        assert args.clear_raw_logs is False, "To use clear_raw_logs, you must set the summarization argument."
    if args.summarization and args.summ_period is None:
        print_system_log("SUMMARIZATION WITHOUT PERIOD WILL IGNORE ALL OTHER SETTINGS FOR PROMPT. THE WHOLE CHAT LOGS WILL BE SUMMARIZED INTO A PROMPT.")
    else:
        assert args.concat_policy in ['simple', 'retrieval'], "The concatenation policy should be either 'simple' or 'retrieval'."
        if args.max_num_msgs is None:
            print_system_log("ANY CONCATENATION POLICY WITH NO SPECIFIC MAX NUMBER OF MESSAGES WOULD BE CASTED INTO THE SIMPLE CONCATENATION.")
            args.concat_policy = 'simple'  # The retrieval concatenation without any number of turns is not different from the simple concatenation.
    if args.generate_states:
        print_system_log("YOU SET update_state=True WHICH AUTOMATICALLY TURNS OFF include_functions.")
        args.include_functions = False

    api_key = input("Enter the API key for OpenAI API: ")
    os.environ['OPENAI_API_KEY'] = api_key
    log_break()
    engine = OpenAIEngine(model=args.model_idx)

    # Loading the unit tests.
    with open(args.tests_path, 'r') as f:
        unit_tests = json.load(f)
    test_results = []

    async def main():
        num_tests = len(unit_tests)
        score_sum = 0.0
        for u, unit_test in enumerate(unit_tests):
            print('-' * 100)
            print(f"Testing case {u+1}...")
            res = await test(args, engine, unit_test)
            score_sum += res['score']
            test_results.append(res)

            time.sleep(30)

        await engine.close()

        # Exporting the result.
        if not os.path.isdir(args.result_dir):
            os.makedirs(args.result_dir)
        with open(f"{args.result_dir}/unit-tests-time={execution_time}.json", 'w') as f:
            json.dump(test_results, f)

        print(f"Score: {score_sum} / {num_tests * 100}")

    asyncio.run(main())
