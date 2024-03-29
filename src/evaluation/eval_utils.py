from inputimeout import inputimeout
from typing import Any, List
from eval_constants import RULE_SUMMARY

import logging
import json

log = logging.getLogger("kani")
message_log = logging.getLogger("kani.messages")


# Trivial definitions for CLI printing.
def print_logic_start(title: str):
    print('#' * 100)
    print(f"{title.upper()}")


def print_question_start():
    print('-' * 100)


def print_system_log(msg: str, after_break: bool=False):
    print(f"[SYSTEM] {msg}")
    if after_break:
        log_break()

def get_player_input(name: str=None, per_player_time: int=None, after_break: bool=False):
    if name is None:
        query = input("INPUT: ")
    else:
        query = inputimeout(f"[PLAYER] {name.replace('_', ' ')}: ", timeout=per_player_time)

    if after_break:
        log_break()
        
    return query


def logic_break():
    print('\n')


def log_break():
    print()


# Default function for allowing a multi-choice query.
def select_options(options: List[Any]):
    while True:
        for o, option in enumerate(options):
            print(f"({o+1}) {option}")
        res = get_player_input(after_break=True)

        try:
            res = int(res)
            if res < 1 or res > len(options):
                print_system_log(f"The allowed value should be from {1} tO {len(options)}.", after_break=True)
            else:
                return res-1
        except ValueError:
            print_system_log("The input should be an integer.", after_break=True)


# Default function for allowing a continuous score value.
def give_score(max_score: float, min_score: float):
    while True:
        print_system_log(f"Give the score between {min_score} - {max_score}.")
        score = get_player_input(after_break=True)

        try:
            score = float(score)
            if score < min_score or score > max_score:
                print_system_log(f"The allowed value should be from {min_score} to {max_score}.", after_break=True)
            else:
                return score
        except ValueError:
            print_system_log("The input should be a valid number.", after_break=True)


# Converting a dictionary into a natural language message.
def convert_into_natural(message: dict):
    name, content = message['name'], message['content']
    if message['role'] == 'assistant':
        if name is None:
            name = "[Game Manager]"
        else:
            name = f"[Game Manager] {name.replace('_', ' ')}:"
    if message['role'] == 'user':
        if name is None:
            name = "[Player]"
        else:
            name = f"[Player] {name.replace('_', ' ')}:"
    if message['role'] == 'function':
        name = f"[Function] {name}:"
    if message['role'] == 'system':
        if name is None:
            name = "[SYSTEM]"
        else:
            name = f"[SYSTEM] {name}:"
    
    return f"{name} {content}"


# Converting the scene state into a natural HTML form.
def convert_scene_to_html(scene_state: dict):
    chapter = scene_state['chapter']
    scene = scene_state['scene']
    scene_summary = scene_state['scene_summary']
    npcs = scene_state['npcs']
    success_condition = scene_state['success_condition']
    failure_condition = scene_state['failure_condition']
    game_flow = scene_state['game_flow']
    environment = scene_state['environment']
    random_tables = scene_state['random_tables']
    consequences = scene_state['consequences']
    is_action_scene = scene_state['is_action_scene']

    def get_npc(npc: dict):
        return f"Kin: {npc['kin']}<br>Persona: {', '.join(npc['persona'])}<br>Goal: {npc['goal']}<br>Trait: {npc['trait']}<br>Flaw: {npc['flaw']}"
    
    def get_table(table: list):
        return '<br>'.join(table)

    res = []
    res.append(f"<strong><u>Chapter</u><strong>: {chapter}")
    res.append(f"<strong><u>Scene</u><strong>: {scene}")
    res.append(f"<strong><u>Scene Summary</u><strong>: <br>{'<br>'.join(scene_summary)}")

    npc_blocks = []
    for i, (name, npc) in enumerate(npcs.items()):
        npc_blocks.append(f"[{name}]<br> {get_npc(npc)}")
    res.append(f"<strong><u>NPCs</u><strong>: <br>{'<br>'.join(npc_blocks)}")

    res.append(f"<strong><u>Success Condition</u><strong>: {success_condition}")
    res.append(f"<strong><u>Failure Condition</u><strong>: {failure_condition}")

    game_flow = [f"({f+1}) {flow}" for f, flow in enumerate(game_flow)]
    res.append(f"<strong><u>Game Flow</u><strong>: <br>{'<br>'.join(game_flow)}")

    environment = [f"[{name}] {desc}" for name, desc in environment.items()]
    res.append(f"<strong><u>Environment</u><strong>: <br>{'<br>'.join(environment)}")

    random_table_blocks = []
    for name, entries in random_tables.items():
        random_table_blocks.append(f"[{name}] {get_table(entries)}")
    res.append(f"<strong><u>Random Tables</u><strong>: <br>{'<br>'.join(random_table_blocks)}")

    res.append(f"<strong><u>Consequences</u><strong>: {consequences}")
    res.append(f"<strong><u>Action Scene</u><strong>: {is_action_scene}")

    return '<br><br>'.join(res)


# Converting the player state into a natural HTML form.
def convert_player_to_html(player_state: dict):
    kin = player_state['kin']
    persona = player_state['persona']
    goal = player_state['goal']
    traits = player_state['traits']
    flaws = player_state['flaws']
    inventory = player_state['inventory']
    additional_notes = player_state['additional_notes']

    res = []
    res.append(f"<strong><u>Kin</u><strong>: {kin}")
    res.append(f"<strong><u>Persona</u><strong>: <br>{'<br>'.join(persona)}")
    res.append(f"<strong><u>Goal</u><strong>: {goal}")
    
    traits = [f"[{name}] {desc}" for name, desc in traits.items()]
    res.append(f"<strong><u>Traits</u><strong>: <br>{'<br>'.join(traits)}")

    flaws = [f"[{name}] {desc}" for name, desc in flaws.items()]
    res.append(f"<strong><u>Flaws</u><strong>: <br>{'<br>'.join(flaws)}")

    inventory = [f"[{name}] {desc}" for name, desc in inventory.items()]
    res.append(f"<strong><u>Inventory</u><strong>: <br>{'<br>'.join(inventory)}")

    res.append(f"<strong><u>Additional Notes</u><strong>: <br>{'<br>'.join(additional_notes)}")

    return '<br><br>'.join(res)


# A sub-logic for showing the scene state.
def show_scene_state(scene: dict):
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
            break

# A sub-logic for showing the player states.
def show_player_states(players: list[dict]):
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
            break

# A sub-logic for showing the past chat history.
def show_past_history(past_history: list[dict]):
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
            break


# A sub-logic for showing the current queries.
def show_current_queries(current_queries: list[dict]):
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
            break


# A sub-logic for showing the game rules.
def show_game_rules():
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
            break
