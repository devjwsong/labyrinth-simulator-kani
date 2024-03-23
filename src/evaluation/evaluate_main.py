from copy import deepcopy
from eval_utils import print_system_log, print_question_start, get_player_input, log_break, select_options, give_score
from eval_constants import (
    MAIN_OPTIONS,
    STATE_CONSISTENCY_RUBRIC,
    RULE_SUMMARY
)

import json
import argparse
import os


# The main evaluation logic for responses.
def response_evaluation_logic(data: dict):
    def count_target_responses(data):
        count = 0
        for gen in data[:-1]:
            generated = gen['generated']
            if generated['content'] is not None:
                count += 1
        return count

    num_targets = count_target_responses(data)
    cur = 1
    scored = deepcopy(data)
    for gen in scored[:-1]:  # One session is per one generation from the model.
        scene = gen['scene']
        players = gen['players']
        past_history = gen['past_history']
        current_queries = gen['current_queries']
        generated = gen['generated']  # Function result is ignored.

        # The response should have non-null content.
        if generated['content'] is None:
            continue

        gen['response_scores'] = {}
        while True:
            print_system_log(f"Evaluation target: {cur} / {num_targets}:", after_break=True)
            print(generated['content'])
            log_break()
            
            # 1. Consistency with the current progress.
            print_system_log(STATE_CONSISTENCY_RUBRIC['question'], after_break=True)
            print_system_log("You should consider:")
            for inst in STATE_CONSISTENCY_RUBRIC['specifications']:
                print(inst)
                details = STATE_CONSISTENCY_RUBRIC['specifications'][inst]
                for detail in details:
                    print(f"--- {detail}")
            log_break()
            
            selected = select_options(MAIN_OPTIONS)
            if selected == 0:  # Show the scene state.
                pass
            elif selected == 1:  # Show the player states.
                pass
            elif selected == 2:  # Show the past chat history.
                pass
            elif selected == 3:  # Show the current queries.
                pass
            elif selected == 4:  # Show the game rules.
                pass
            else:  # Give the score.
                score = give_score(STATE_CONSISTENCY_RUBRIC['max_score'], STATE_CONSISTENCY_RUBRIC['min_score'])
                gen['response_scores']['consistency'] = score
                break

        cur += 1

    assert cur == num_targets + 1, "Inconsistency occured while counting the targets."

    return scored


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--game_file', type=str, required=True, help="The path of the gameplay record file to evaluate.")
    parser.add_argument('--eval_focus', type=str, required=True, help="The evaluation focus: Response or Function?")
    
    args = parser.parse_args()

    assert args.eval_focus in ['response', 'function'], "The evaluation focus should be either 'response' or 'function'."

    print_question_start()
    print_system_log("BEFORE STARTING THE GAME, GIVE US YOUR USERNAME, WHICH IS USED FOR RECORDING PURPOSE.")
    username = get_player_input(after_break=True)

    # Loading the gameplay data file.
    with open(args.game_file, 'r') as f:
        data = json.load(f)

    if args.eval_focus == 'response':
        scored = response_evaluation_logic(data)

    # Export the scored data.
    game_file_dir, file_name = '/'.join(args.game_file.split('/')[:-1]), args.game_file.split('/')[-1].replace('.json', '')
    evaluation_file_dir = f"evaluated_by_{username}/{game_file_dir}"
    if not os.path.isdir(evaluation_file_dir):
        os.makedirs(evaluation_file_dir)

    with open(f"{evaluation_file_dir}/{file_name}.json", 'w') as f:
        json.dump(scored, f)
