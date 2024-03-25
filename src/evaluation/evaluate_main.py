from eval_utils import (
    print_system_log, 
    print_question_start, 
    print_logic_start, 
    get_player_input, 
    log_break, 
    select_options, 
    give_score,
    show_game_rules,
    show_scene_state,
    show_player_states,
    show_past_history,
    show_current_queries
)
from eval_constants import (
    RESPONSE_CONSISTENCY_OPTIONS,
    RESPONSE_RELIABILITY_OPTIONS,
    RESPONSE_INTERESTINGNESS_OPTIONS,
    CONSISTENCY_RUBRIC,
    RELIABILITY_RUBRIC,
    INTERESTINGNESS_RUBRIC
)

import json
import argparse
import os


# A sub-logic for each target response.
def target_response_logic(initial_scene: dict, initial_players: list[dict], gen: dict, metric: str):
    past_history = gen['past_history']
    current_queries = gen['current_queries']

    if metric == 'consistency':
        max_score, min_score = CONSISTENCY_RUBRIC['max_score'], CONSISTENCY_RUBRIC['min_score']
        idx = select_options(RESPONSE_CONSISTENCY_OPTIONS)
        
        if idx == 0:
            show_scene_state(initial_scene)
            return False
        
        elif idx == 1:
            show_player_states(initial_players)
            return False

        elif idx == 2:
            show_past_history(past_history)
            return False

        elif idx == 3:
            show_current_queries(current_queries)
            return False

        elif idx == 4:
            gen['response_scores'][metric] = {}
            score = give_score(max_score, min_score)
            comment = input("Leave a comment that explains why you chose to give the score: ")
            gen['response_scores'][metric] = {
                'score': score,
                'comment': comment
            }
            return True

    elif metric == 'reliability':
        max_score, min_score = RELIABILITY_RUBRIC['max_score'], RELIABILITY_RUBRIC['min_score']
        idx = select_options(RESPONSE_RELIABILITY_OPTIONS)

        if idx == 0:
            show_game_rules()
            return False
        
        elif idx == 1:
            show_scene_state(initial_scene)
            return False
        
        elif idx == 2:
            show_player_states(initial_players)
            return False

        elif idx == 3:
            show_past_history(past_history)
            return False

        elif idx == 4:
            show_current_queries(current_queries)
            return False

        elif idx == 5:
            gen['response_scores'][metric] = {}
            score = give_score(max_score, min_score)
            comment = input("Leave a comment that explains why you chose to give the score: ")
            gen['response_scores'][metric] = {
                'score': score,
                'comment': comment
            }
            return True

    elif metric == 'interestingness':
        max_score, min_score = INTERESTINGNESS_RUBRIC['max_score'], INTERESTINGNESS_RUBRIC['min_score']
        idx = select_options(RESPONSE_INTERESTINGNESS_OPTIONS)

        gen['response_scores'][metric] = {}
        score = give_score(max_score, min_score)
        comment = input("Leave a comment that explains why you chose to give the score: ")
        gen['response_scores'][metric] = {
            'score': score,
            'comment': comment
        }
        return True


# The main evaluation logic for responses.
def response_evaluation_logic(data: dict):
    initial_scene = data[0]['scene']
    initial_players = data[0]['players']

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
            to_next_eval = target_response_logic(initial_scene, initial_players, gen, metric='consistency')
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
            to_next_eval = target_response_logic(initial_scene, initial_players, gen, metric='reliability')
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
            to_next_eval = target_response_logic(initial_scene, initial_players, gen, metric='interestingness')
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
