from utils import print_question_start, print_system_log, get_player_input, print_logic_start, logic_break, select_options
from datetime import datetime
from pytz import timezone
from argparse import Namespace

import argparse
import json
import os


# Exporting the created player characters.
def export_result(result: list[dict], username: str, execution_time: str):
    num_players = len(result)

    file_dir = f"players/num_players={num_players}"
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    file_path = f"{file_dir}/{username}-time={execution_time}.json"
    with open(file_path, 'w') as f:
        json.dump(result, f)


# Creating one player character
def create_player(character_data: dict):
    print_system_log("IN THE LABYRINTH, THERE ARE MULTIPLE KINS TO CHOOSE.")
    print_system_log("EACH KIN HAS UNIQUE PERSONA AND EVERY CHARACTER HAS TRAITS AND FLAWS WHICH MIGHT AFFECT YOUR GAMEPLAY.")
    print_system_log("EACH CHARACTER CAN SELECT ONE TRAIT AND ONE FLAW, BUT THE NUMBER OF TRAITS/FLAWS MIGHT VARY DEPENDING ON YOUR KIN.", after_break=True)
    while True:
        # Showing the list of kins.
        print_system_log("SELECT THE KIN TO SEE MORE DETAIL.")
        kins = list(character_data['kins'].keys())
        selected = select_options(kins)
        kin = kins[selected]
        info = character_data['kins'][kin]
        persona = info['persona']
        guide = info['guide']
        additional_notes = info['additional_notes'] if 'additional_notes' in info else []

        # Showing the details of the selected kin.
        print_system_log("INFORMATION ON THE SELECTED KIN IS...")
        print(f"Kin: {kin}")
        for s, sent in enumerate(persona):
            print(f"({s+1}) {sent}")
        print(' '.join(guide))

        # Confirming the kin.
        print_question_start()
        print_system_log(f"ARE YOU GOING TO GO WITH {kin}?")
        selected = select_options(['Yes', 'No'])
        if selected == 1:
            print_system_log("GOING BACK TO THE LIST...")
            continue

        player = {
            'name': "",
            'kin': kin,
            'persona': persona,
            'goal': "",
            'traits': {},
            'flaws': {},
            'inventory': {},
            'additional_notes': additional_notes
        }

        # Setting the name.
        print_question_start()
        print_system_log("WHAT IS YOUR NAME?")
        name = get_player_input(after_break=True)

        # Removing the white space in the name.
        player['name'] = name.replace(' ', '_')

        # Setting the goal.
        print_question_start()
        print_system_log("WHAT IS YOUR GOAL? WHY DID YOU COME TO THE LABYRINTH TO CHALLENGE THE GOBLIN KING?")
        player['goal'] = get_player_input(after_break=True)

        # Setting the character-specific additional features.
        if kin == 'Dwarf':
            print_question_start()
            print_system_log("YOU'VE SELECTED DWARF. SELECT YOUR JOB.")
            jobs_and_tools = info['tables']['jobs_and_tools']
            selected = select_options(jobs_and_tools)
            job_and_tool = jobs_and_tools[selected]
            player['traits']['Job'] = f"My job is {job_and_tool['job']} and I can use my {job_and_tool['tool']} professionally."

            print_question_start()
            print_system_log(f"GIVE MORE DETAILS ON YOUR TOOL: {job_and_tool['tool']}")
            item_description = get_player_input(after_break=True)
            player['inventory'][job_and_tool['tool']] = item_description

        elif kin == 'Firey' or kin == 'Knight of Yore' or kin == 'Worm':
            player['traits'].update(info['default_traits'])

        elif kin == 'Goblin':
            print_question_start()
            print_system_log("YOU'VE SELECTED GOBLIN. SPECIFY WHY YOU ARE AGAINST THE GOBLIN KING.")
            default_traits = info['default_traits']
            reason = get_player_input(after_break=True)
            if len(reason) > 0:
                default_traits['Goblin feature'] = reason
            player['traits'].update(default_traits)

        elif kin == 'Horned Beast':
            print_question_start()
            print_system_log("YOU'VE SELECTED HORNED BEAST. SELECT ONE OBJECT TYPE YOU CAN CONTROL.")
            objects = info['tables']['objects']
            selected = select_options(objects)
            player['traits']['Control object'] = f"I can control an object of type {objects[selected]}."
            player['flaws'].update(info['default_flaws'])

        # Picking up a trait.
        print_question_start()
        print_system_log("NOW, SELECT ONE TRAIT FROM THE GIVEN LIST.")
        cands = character_data['traits']
        selected = select_options(cands)
        player['traits'][cands[selected]['trait']] = cands[selected]['description']
        if kin == 'Human':
            extra_cands = [entry for entry in character_data['traits'] if entry['trait'] not in player['traits']]
            print_question_start()
            print_system_log("YOU'VE SELECTED HUMAN. YOU CAN PICK ONE MORE EXTRA TRAIT.")
            selected = select_options(extra_cands)
            player['traits'][extra_cands[selected]['trait']] = extra_cands[selected]['description']
            
        # Picking up a flaw.
        print_question_start()
        print_system_log("NEXT, SELECT ONE FLAW FROM THE GIVEN LIST.")
        filtered_flaws = []
        for entry in character_data['flaws']:
            included = True
            if 'restriction' in entry:
                for trait in player['traits']:
                    if trait.startswith(entry['restriction']):
                        included = False
                        break
            if included:
                filtered_flaws.append(entry)
        selected = select_options(filtered_flaws)
        player['flaws'][filtered_flaws[selected]['flaw']] = filtered_flaws[selected]['description']

        print_question_start()
        print_system_log("FINALLY, CONFIRM IF THESE SPECIFICATIONS ARE MATCHED WITH YOUR CHOICES.")
        print(player)

        print_question_start()
        print_system_log("ARE THEY CORRECT?")
        selected = select_options(['Yes', 'No'])
        if selected == 1:
            print_system_log("GOING BACK TO THE LIST...", after_break=True)
            continue

        print_system_log(f"THE PLAYER CHARACTER {player['name']} HAS BEEN CREATED SUCCESSFULLY.")
        return player


# The main logic for player character creation.
def create_players(args: Namespace):
    with open("data/characters.json", 'r') as f:
        character_data = json.load(f)
    players = []

    for p in range(args.num_players):
        print_logic_start(f"LET'S CREATE THE PLAYER {p+1}.")
        player = create_player(character_data)
        players.append(player)
        logic_break()

    return players


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_players', type=int, default=1, help="The number of player characters to create.")

    args = parser.parse_args()

    print_question_start()
    print_system_log("GIVE US YOUR USERNAME, WHICH IS USED FOR RECORDING PURPOSE.")
    username = get_player_input(after_break=True)

    now = datetime.now(timezone('US/Eastern'))
    execution_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    # Running the main logic.
    players = create_players(args)

    # Exporting the result
    export_result(players, username, execution_time)
