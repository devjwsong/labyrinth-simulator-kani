from eval_utils import print_system_log, print_question_start, print_logic_start, get_player_input, log_break, select_options, give_score
from eval_constants import (
    MAIN_OPTIONS,
    CONSISTENCY_RUBRIC,
    RELIABILITY_RUBRIC,
    INTERESTINGNESS_RUBRIC,
    RULE_SUMMARY
)

import json
import argparse
import os


def target_logic(gen: dict, metric: str):
    if metric == 'consistency':
        max_score, min_score = CONSISTENCY_RUBRIC['max_score'], CONSISTENCY_RUBRIC['min_score']
    elif metric == 'reliability':
        max_score, min_score = RELIABILITY_RUBRIC['max_score'], RELIABILITY_RUBRIC['min_score']
    elif metric == 'interestingness':
        max_score, min_score = INTERESTINGNESS_RUBRIC['max_score'], INTERESTINGNESS_RUBRIC['min_score']

    scene = gen['scene']
    players = gen['players']
    past_history = gen['past_history']
    current_queries = gen['current_queries']

    selected = select_options(MAIN_OPTIONS)
    if selected == 0:  # Show the scene state.
        keys = list(scene.keys())
        k = 0
        while k < len(keys):
            options = []
            if k == 0:
                options = ["Next", "Go back to the evaluation"]
            elif k == len(keys)-1:
                options = ["Prev", "Go back to the evaluation"]
            else:
                options = ["Next", "Prev", "Go back to the evaluation"]
            
            key = keys[k]
            print_question_start()
            print_system_log(f"{key}: ")
            print(json.dumps(scene[key], indent=4))
            log_break()

            idx = select_options(options)
            if options[idx] == "Next":
                k += 1
            elif options[idx] == "Prev":
                k -= 1
            else:
                return False

    elif selected == 1:  # Show the player states.
        num_players = len(players)
        p = 0
        while p < num_players:
            if p == 0:
                options = ["Next", "Go back to the evaluation"]
            elif p == num_players-1:
                options = ["Prev", "Go back to the evaluation"]
            else:
                options = ["Next", "Prev", "Go back to the evaluation"]

            player = players[p]
            print_question_start()
            print_system_log(f"Player {p+1}: {player['name']}")
            print(json.dumps(player, indent=4))
            log_break()

            idx = select_options(options)
            if options[idx] == "Next":
                p += 1
            elif options[idx] == "Prev":
                p -= 1
            else:
                return False

    elif selected == 2:  # Show the past chat history.
        num_messages = len(past_history)
        m = 0
        while m < num_messages:
            if m == 0:
                options = ["Next", "Go back to the evaluation"]
            elif m == num_messages-1:
                options = ["Prev", "Go back to the evaluation"]
            else:
                options = ["Next", "Prev", "Go back to the evaluation"]
            
            message = past_history[m]
            print_question_start()
            print(json.dumps(message, indent=4))
            log_break()

            idx = select_options(options)
            if options[idx] == "Next":
                m += 1
            elif options[idx] == "Prev":
                m -= 1
            else:
                return False

    elif selected == 3:  # Show the current queries.
        num_messages = len(current_queries)
        m = 0
        while m < num_messages:
            if m == 0:
                options = ["Next", "Go back to the evaluation"]
            elif m == num_messages-1:
                options = ["Prev", "Go back to the evaluation"]
            else:
                options = ["Next", "Prev", "Go back to the evaluation"]
            
            message = current_queries[m]
            print_question_start()
            print(json.dumps(message, indent=4))
            log_break()

            idx = select_options(options)
            if options[idx] == "Next":
                m += 1
            elif options[idx] == "Prev":
                m -= 1
            else:
                return False

    elif selected == 4:  # Show the game rules.
        num_parts = len(RULE_SUMMARY)
        r = 0
        while r < num_parts:
            if r == 0:
                options = ["Next", "Go back to the evaluation"]
            elif r == num_parts-1:
                options = ["Prev", "Go back to the evaluation"]
            else:
                options = ["Next", "Prev", "Go back to the evaluation"]

            part = RULE_SUMMARY[r]
            print_question_start()
            print_system_log(f"Labyrinth's rule ({r+1})")
            print('\n'.join(part))
            log_break()

            idx = select_options(options)
            if options[idx] == "Next":
                r += 1
            elif options[idx] == "Prev":
                r -= 1
            else:
                return False

    else:  # Give the score.
        score = give_score(max_score, min_score)
        gen['response_scores'][metric] = score
        return True


# The main evaluation logic for responses.
def response_evaluation_logic(data: dict):
    def extract_target_response(data):
        scored = []
        for gen in data[:-1]:
            generated = gen['generated']
            if generated['content'] is not None:
                scored.append(gen)
        return scored

    scored = extract_target_response(data)
    for g, gen in enumerate(scored):  # One session is per one generation from the model.
        generated = gen['generated']  # Function result is ignored.

        # The response should have non-null content.
        if generated['content'] is None:
            continue

        gen['response_scores'] = {}
        while True:
            print_system_log(f"Evaluation target: {g+1} / {len(scored)}:", after_break=True)
            
            # 1. Consistency with the current progress.
            print_logic_start("Target Response: ")
            print(generated['content'])
            log_break()
            
            print_question_start()
            print_system_log(f"QUESTION: {CONSISTENCY_RUBRIC['question']}", after_break=True)
            print_system_log("You should consider:")
            for i, inst in enumerate(CONSISTENCY_RUBRIC['specifications']):
                print(f"({i+1}) {inst}")
                details = CONSISTENCY_RUBRIC['specifications'][inst]
                for detail in details:
                    print(f"-> {detail}")
            log_break()

            print_question_start()
            to_next_eval = target_logic(gen, metric='consistency')
            if to_next_eval:
                break

        while True:
            # 2. Reliability as a game manager.
            print_logic_start("Target Response: ")
            print(generated['content'])
            log_break()

            print_question_start()
            print_system_log(f"QUESTION: {RELIABILITY_RUBRIC['question']}", after_break=True)
            print_system_log("You should consider:")
            for i, inst in enumerate(RELIABILITY_RUBRIC['specifications']):
                print(f"({i+1}) {inst}")
                details = RELIABILITY_RUBRIC['specifications'][inst]
                for detail in details:
                    print(f"-> {detail}")
            log_break()

            print_question_start()
            to_next_eval = target_logic(gen, metric='reliability')
            if to_next_eval:
                break

        while True:
            # 3. Interestingness of the response.
            print_logic_start("Target Response: ")
            print(generated['content'])
            log_break()

            print_question_start()
            print_system_log(f"QUESTION: {INTERESTINGNESS_RUBRIC['question']}", after_break=True)
            print_system_log("You should consider:")
            for i, inst in enumerate(INTERESTINGNESS_RUBRIC['specifications']):
                print(f"({i+1}) {inst}")
                details = INTERESTINGNESS_RUBRIC['specifications'][inst]
                for detail in details:
                    print(f"-> {detail}")
            log_break()

            print_question_start()
            to_next_eval = target_logic(gen, metric='interestingness')
            if to_next_eval:
                break

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
    game_file_dir, file_name = '/'.join(args.game_file.split('/')[1:-1]), args.game_file.split('/')[-1].replace('.json', '')
    evaluation_file_dir = f"evaluated_by_{username}/{game_file_dir}"
    if not os.path.isdir(evaluation_file_dir):
        os.makedirs(evaluation_file_dir)

    with open(f"{evaluation_file_dir}/{file_name}.json", 'w') as f:
        json.dump(scored, f)
