import os
import sys

cur_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(cur_dir, '..'))
sys.path.insert(0, src_path)

from kani import Kani
from kani.engines.openai import OpenAIEngine
from constants import (
    CONSISTENCY_RUBRIC,
    RELIABILITY_RUBRIC,
    INTERESTINGNESS_RUBRIC,
)
from utils import (
    print_question_start,
    print_system_log,
    get_player_input,
    log_break
)

import json
import argparse
import os


# Main evaluation logic.
def evaluate():
    pass


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

    # Creating the evaluator kani.
    api_key = input("Enter the API key for OpenAI API: ")
    os.environ['OPENAI_API_KEY'] = api_key
    log_break()
    engine = OpenAIEngine(model=args.model_idx)

    scored = evaluate()

    # Export the scored data.
    game_file_dir, file_name = '/'.join(args.game_file.split('/')[1:-1]), args.game_file.split('/')[-1].replace('.json', '')
    evaluation_file_dir = f"evaluated_by_{username}/{game_file_dir}"
    if not os.path.isdir(evaluation_file_dir):
        os.makedirs(evaluation_file_dir)

    with open(f"{evaluation_file_dir}/{file_name}.json", 'w') as f:
        json.dump(scored, f)
