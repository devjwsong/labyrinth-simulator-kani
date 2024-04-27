import os
import sys
from time import sleep

cur_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(cur_dir, '..'))
sys.path.insert(0, src_path)

from kani import Kani
from kani.engines.openai import OpenAIEngine
from kani.models import ChatMessage
from copy import deepcopy
from tqdm import tqdm
from constants import (
    EVALUATOR_INSTRUCTION,
    RULE_SUMMARY,
    CONSISTENCY_RUBRIC,
    RELIABILITY_RUBRIC,
    INTERESTINGNESS_RUBRIC,
)
from utils import (
    logic_break,
    print_question_start,
    print_system_log,
    get_player_input,
    log_break,
    clean_logs,
    convert_into_message,
    convert_into_number
)

import json
import argparse
import os
import asyncio
import re

MAX_NUM_TARGETS = 10


# Filtering only the target responses.
def extract_target_response(data):
    target_objs = []
    for gen in data[:-1]:
        generated = gen['generated']
        if generated['content'] is not None:
            target_objs.append(gen)
    return target_objs


def convert_into_score(res):
    regex = '[+-]?[0-9]+\.[0-9]+'
    matches = re.findall(regex, res)
    if matches:
        return float(matches[0])
    return None


# Making a rubric into a natural lanauge description.
def convert_rubric_into_queries(rubric: dict):
    query1 = rubric['question']
    query1 = f"{query1}\n\nIn more details you need to check:"
    for s, (spec, details) in enumerate(rubric['specifications'].items()):
        spec_desc = spec
        if len(details) > 0:
            spec_desc = f"{spec_desc} ({' '.join(details)})"
        query1 += f"\n({s+1}) {spec_desc}"
    if len(rubric['notes']) > 0:
        query1 = f"{query1}\n\nNote that: {' '.join(rubric['notes'])}"
    query1 += f"\n\nFirst, explain what is good and bad in this response as detailed as possible."

    query2 = f"Based on the explanation, give a score between {rubric['min_score']} and {rubric['max_score']}."
    query2 = f"{query2}\n\nYou should be strict when giving a score."

    return query1, query2


# Main evaluation logic.
def evaluate(engine: OpenAIEngine, data: dict):
    initial_scene_state, initial_player_states = data[0]['scene'], data[0]['players']
    target_objs = extract_target_response(data)[:MAX_NUM_TARGETS]
    consistency_query1, consistency_query2 = convert_rubric_into_queries(CONSISTENCY_RUBRIC)
    reliability_query1, reliability_query2 = convert_rubric_into_queries(RELIABILITY_RUBRIC)
    interest_query1, interest_query2 = convert_rubric_into_queries(INTERESTINGNESS_RUBRIC)
    scored = []
    
    async def logic():
        generation_params = {
            'temperature': 0.5,
            'top_p': 1.0,
        }

        # Processing the targets.
        for o, obj in enumerate(tqdm(target_objs)):
            print("#" * 100)
            past_history, current_queries, generated = obj['past_history'], obj['current_queries'], obj['generated']
            obj['scores'] = {}
            
            # Combining the current queries into the past chat history.
            chat_history = past_history + clean_logs(current_queries)
            chat_messages = [convert_into_message(hist) for hist in chat_history]

            # Setting Kani agent.
            system_prompt = ' '.join(EVALUATOR_INSTRUCTION)
            rule_content = '\n'.join([' '.join(part) for part in RULE_SUMMARY])
            rule_prompt = ChatMessage.system(name="Rule", content=rule_content)
            scene_prompt = ChatMessage.system(name="Initial_Scene_state", content=str(initial_scene_state))
            player_prompts = []
            for player_state in initial_player_states:
                player_prompt = ChatMessage.system(name="Initial_Player_state", content=str(player_state))
                player_prompts.append(player_prompt)
            kani = Kani(
                engine=engine,
                system_prompt=system_prompt, 
                always_included_messages=[rule_prompt, scene_prompt] + player_prompts, 
                chat_history=deepcopy(chat_messages)
            )

            # 1. Consistency.
            print('@' * 20 + "Consistency" + '@' * 20)
            query = f"Target response: {generated['content']}\n\n{consistency_query1}"
            res = await kani.chat_round_str(query, **generation_params)
            print(res)
            query = consistency_query2
            res = await kani.chat_round_str(query, **generation_params)
            print(res)
            score = convert_into_score(res)
            if score is None:
                score = convert_into_number(res)
            obj['scores']['consistency'] = score
            kani.chat_history = deepcopy(chat_messages)

            # 2. Reliability
            print('@' * 20 + "Reliability" + '@' * 20)
            query = f"Target response: {generated['content']}\n\n{reliability_query1}"
            res = await kani.chat_round_str(query, **generation_params)
            print(res)
            query = reliability_query2
            res = await kani.chat_round_str(query, **generation_params)
            print(res)
            score = convert_into_score(res)
            if score is None:
                score = convert_into_number(res)
            obj['scores']['reliability'] = score
            kani.chat_history = deepcopy(chat_messages)

            # 3. Interest
            print('@' * 20 + "Interest" + '@' * 20)
            query = f"Target response: {generated['content']}\n\n{interest_query1}"
            res = await kani.chat_round_str(query, **generation_params)
            print(res)
            query = interest_query2
            res = await kani.chat_round_str(query, **generation_params)
            print(res)
            score = convert_into_score(res)
            if score is None:
                score = convert_into_number(res)
            obj['scores']['interest'] = score
            kani.chat_history = deepcopy(chat_messages)

            scored.append(obj)

            logic_break()
            sleep(60)

        await engine.close()

    asyncio.run(logic())

    return scored


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--game_file', type=str, required=True, help="The path of the gameplay record file to evaluate.")
    parser.add_argument('--model_idx', type=str, required=True, help="The index of the evaluator model.")
    
    args = parser.parse_args()

    print_question_start()
    print_system_log("BEFORE STARTING THE GAME, GIVE US YOUR USERNAME, WHICH IS USED FOR RECORDING PURPOSE.")
    username = get_player_input(after_break=True)

    # Loading the gameplay data file.
    with open(args.game_file, 'r') as f:
        data = json.load(f)

    # Creating the evaluator enigne.
    api_key = input("Enter the API key for OpenAI API: ")
    os.environ['OPENAI_API_KEY'] = api_key
    log_break()
    engine = OpenAIEngine(model=args.model_idx)
    
    # Main logic.
    scored = evaluate(engine, data)

    # Export the scored data.
    game_file_dir, file_name = '/'.join(args.game_file.split('/')[1:-1]), args.game_file.split('/')[-1].replace('.json', '')
    evaluation_file_dir = f"evaluated_by_{username}/eval_model={args.model_idx}/{game_file_dir}"
    if not os.path.isdir(evaluation_file_dir):
        os.makedirs(evaluation_file_dir)

    with open(f"{evaluation_file_dir}/{file_name}.json", 'w') as f:
        json.dump(scored, f)
