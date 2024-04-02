from eval_constants import (
    CONSISTENCY_RUBRIC,
    INTERESTINGNESS_RUBRIC,
    RELIABILITY_RUBRIC,
    TASK_INTRODUCTION,
)
from eval_utils import (
    convert_into_natural,
    convert_scene_to_html,
    convert_player_to_html,
    clean_history
)

import argparse
import json
import os


# Generating .txt file to import the Qualtrics survey.
def generate_survey(data: list[dict]):
    survey = ["[[AdvancedFormat]]"]

    initial_scene_state, initial_player_states = data[0]['scene'], data[0]['players']
    survey.append("[[Question:DB]]")
    introduction = ''.join(TASK_INTRODUCTION)
    survey.append(introduction)
    survey.append("")

    def extract_target_response(data):
        target_objs = []
        for gen in data[:-1]:
            generated = gen['generated']
            if generated['content'] is not None:
                target_objs.append(gen)
        return target_objs

    target_objs = extract_target_response(data)

    # Processing the targets.
    for o, obj in enumerate(target_objs):
        past_history, current_queries, generated = obj['past_history'], obj['current_queries'], obj['generated']
        
        # Combining the current queries into the past chat history.
        chat_history = past_history + clean_history(current_queries)

        # Converting the components.
        scene_state = convert_scene_to_html(initial_scene_state)
        player_states = [convert_player_to_html(player_state) for player_state in initial_player_states]
        chat_history = [convert_into_natural(hist) for hist in chat_history]

        survey.append(f"[[Block:Block{o+1}]]")

        # Showing the scene state.
        survey.append("[[Question:DB]]")
        survey.append("<p><strong><h2>Starting game state</h2></strong><br>")
        survey.append(scene_state)
        survey.append("")

        # Showing the player states.
        for p, player_state in enumerate(player_states):
            name = initial_player_states[p]['name']
            survey.append("[[Question:DB]]")
            survey.append(f"<strong><h2>Starting player state of {name}</h2></strong><br>")
            survey.append(player_state)
            survey.append("")

        # Showing the chat history.
        survey.append("[[Question:DB]]")
        survey.append("<strong><h2>Chat history so far</h2></strong><br>")
        survey.append('<br>'.join(chat_history))
        survey.append("")

        # Showing the target response.
        survey.append("[[Question:DB]]")
        survey.append("<strong><h2>Target response</h2></strong><br>")
        survey.append(convert_into_natural(generated))
        survey.append("")

        # 1. Resposne consistency.
        survey.append("[[Question:MC:SingleAnswer:Horizontal]]")
        survey.append(f"<strong><h3>{CONSISTENCY_RUBRIC['question']}</h3></strong><br>")
        for s, (spec, sub_specs) in enumerate(CONSISTENCY_RUBRIC['specifications'].items()):
            survey.append(f"{s+1}. {spec}<br>")
            for sub_spec in sub_specs:
                survey.append(f"---- {sub_spec}<br>")
        survey.append("<br>")
        for note in CONSISTENCY_RUBRIC['notes']:
            survey.append(f"- <u>{note}</u><br>")
        survey.append("<br>")
        survey.append(f"({', '.join(CONSISTENCY_RUBRIC['examples'])})<br>")
        survey.append("[[Choices]]")
        cur = CONSISTENCY_RUBRIC['min_score']
        survey.append(f"{cur}")
        for i in range(4):
            cur += 1
            survey.append(f"{cur}")
        survey.append("")
        survey.append("[[Question:TE]]")
        survey.append("Give us a comment that explains why you gave that score.")
        survey.append("")

        # 2. Resposne reliability.
        survey.append("[[Question:MC:SingleAnswer:Horizontal]]")
        survey.append(f"<strong><h3>{RELIABILITY_RUBRIC['question']}</h3></strong><br>")
        for s, (spec, sub_specs) in enumerate(RELIABILITY_RUBRIC['specifications'].items()):
            survey.append(f"{s+1}. {spec}<br>")
            for sub_spec in sub_specs:
                survey.append(f"---- {sub_spec}<br>")
        survey.append("<br>")
        for note in RELIABILITY_RUBRIC['notes']:
            survey.append(f"- <u>{note}</u><br>")
        survey.append("<br>")
        survey.append(f"({', '.join(RELIABILITY_RUBRIC['examples'])})<br>")
        survey.append("[[Choices]]")
        cur = RELIABILITY_RUBRIC['min_score']
        survey.append(f"{cur}")
        for i in range(4):
            cur += 1
            survey.append(f"{cur}")
        survey.append("")
        survey.append("[[Question:TE]]")
        survey.append("Give us a comment that explains why you gave that score.")
        survey.append("")

        # 3. Resposne interestingness.
        survey.append("[[Question:MC:SingleAnswer:Horizontal]]")
        survey.append(f"<strong><h3>{INTERESTINGNESS_RUBRIC['question']}</h3></strong><br>")
        for s, (spec, sub_specs) in enumerate(INTERESTINGNESS_RUBRIC['specifications'].items()):
            survey.append(f"{s+1}. {spec}<br>")
            for sub_spec in sub_specs:
                survey.append(f"---- {sub_spec}<br>")
        survey.append("<br>")
        for note in INTERESTINGNESS_RUBRIC['notes']:
            survey.append(f"- <u>{note}</u><br>")
        survey.append("<br>")
        survey.append(f"({', '.join(INTERESTINGNESS_RUBRIC['examples'])})<br>")
        survey.append("[[Choices]]")
        cur = INTERESTINGNESS_RUBRIC['min_score']
        survey.append(f"{cur}")
        for i in range(4):
            cur += 1
            survey.append(f"{cur}")
        survey.append("")
        survey.append("[[Question:TE]]")
        survey.append("Give us a comment that explains why you gave that score.")
        survey.append("")

        # Next page for next target.
        if o < len(target_objs)-1:
            survey.append("[[PageBreak]]")

    return survey    


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--game_file', type=str, required=True, help="The path of the gameplay record file to evaluate.")

    args = parser.parse_args()

    # Loading the data.
    with open(args.game_file, 'r') as f:
        data = json.load(f)

    # One element per one text line in .txt file.
    lines = generate_survey(data)
    lines = [line + '\n' for line in lines]

    # Exporting the data.
    file_dir = args.game_file.split('/')[1:-1]
    file_dir = "surveys/" + '/'.join(file_dir)
    file_name = args.game_file.split('/')[-1][:-5]
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    with open(f"{file_dir}/{file_name}.txt", 'w') as f:
        f.writelines(lines)
