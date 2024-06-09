from inputimeout import inputimeout
from kani.models import ChatMessage, ChatRole
from typing import Any, List

import logging
import string
import random
import re

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

def print_player_log(msg: str, name:str, after_break: bool=False):
    print(f"[PLAYER] {name.replace('_', ' ')}: {msg}")
    if after_break:
        log_break()


def print_manager_log(msg: str, after_break: bool=False):
    print(f"[GAME MANAGER] Goblin King: {msg}")
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
                print_system_log(f"THE ALLOWED VALUE IS FROM {1} TO {len(options)}.", after_break=True)
            else:
                return res-1
        except ValueError:
            print_system_log("THE INPUT SHOULD BE AN INTEGER.", after_break=True)
            

# Function for a randomized/automated multi-choice query. (For simulation)
def select_random_options(options: List[Any]):
    idxs = list(range(len(options)))
    selected = random.choice(idxs)
    return selected

# Removing function-related messages in the messages.
def clean_history(messages: list[ChatMessage]) -> list[ChatMessage]:
    chat_messages = []
    for message in messages:
        if message.role == ChatRole.USER:
            chat_messages.append(message)
        if message.role == ChatRole.ASSISTANT and message.content:
            chat_messages.append(ChatMessage.assistant(name="Goblin_King", content=message.content))
    return chat_messages


# Removing function-related messages in the messages.
def clean_logs(messages: list[dict]) -> list[dict]:
    chat_messages = []
    for message in messages:
        if message['role'] == 'user':
            chat_messages.append(message)
        if message['role'] == 'assistant' and message['content'] is not None:
            message['name'] = "Goblin_King"
            chat_messages.append(message)
    return chat_messages


# Converting ChatMessage into a dictionary.
def convert_into_dict(message: ChatMessage):
    res = {
        'role': message.role.value,
        'name': message.name,
        'content': message.content,
    }
    if message.role == ChatRole.ASSISTANT:
        res['function_call'] = True if message.tool_calls else False
    return res

# Converting a dictionary into ChatMessage.
def convert_into_message(obj: dict):
    if obj['role'] == 'user':
        return ChatMessage.user(name=obj['name'], content=obj['content'])
    if obj['role'] == 'assistant':
        return ChatMessage.assistant(name=obj['name'], content=obj['content'])
    if obj['role'] == 'function':
        return ChatMessage.function(name=obj['name'], content=obj['content'])
    if obj['role'] == 'system':
        return ChatMessage.system(name=obj['name'], content=obj['content'])


# Converting ChatMessage into a natural language message.
def convert_into_natural(message: ChatMessage):
    name, content = message.name, message.content
    if message.role == ChatRole.ASSISTANT:
        if name is None:
            name = "[Game Manager]"
        else:
            name = f"[Game Manager] {name.replace('_', ' ')}:"
    if message.role == ChatRole.USER:
        if name is None:
            name = "[Player]"
        else:
            name = f"[Player] {name.replace('_', ' ')}:"
    if message.role == ChatRole.FUNCTION:
        name = f"[Function] {name}"
    if message.role == ChatRole.SYSTEM:
        if name is None:
            name = "[SYSTEM]"
        else:
            name = f"[SYSTEM] {name}:"
    
    return f"{name} {content}"


# Converting the model response into a number.
def convert_into_number(res: str):
    pattern = r'\d+'
    matches = re.findall(pattern, res)
    if matches:
        return int(matches[0])
    
    return None


# Extracting the class index in the output of a classification problem.
def convert_into_class_idx(res: str, options: list):
    num = convert_into_number(res)
    if num is None or num >= len(options):
        return select_random_options(options)

    return num


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
        random_table_blocks.append(f"[{name}]<br>{get_table(entries)}")
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
