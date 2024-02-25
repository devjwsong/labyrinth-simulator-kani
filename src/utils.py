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


# Removing unnecessary punctuations from the object name.
def remove_punctuation(word: str):
    puncs = list(string.punctuation)
    cut_idx = len(word)
    for i in range(len(word)-1, -1, -1):
        if word[i] in puncs:
            cut_idx = i
        else:
            break
    word = word[:cut_idx]
    return word


# Finding the valid current queries.
def find_current_point(chat_history: list[ChatMessage]) -> int:
    idx = len(chat_history)
    for i in range(len(chat_history)-1, -1, -1):
        if chat_history[i].role == ChatRole.USER and (i == 0 or chat_history[i-1].role != ChatRole.USER):
            idx = i
            break
    return idx


# Converting ChatMessage into a dictionary.
def convert_into_dict(message: ChatMessage):
    res = {
        'role': message.role.value,
        'name': message.name,
        'content': message.content,
    }
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


# Extracting the class index in the output of a classification problem.
def convert_into_class_idx(res: str, options: list):
    pattern = r'\d+'
    matches = re.findall(pattern, res)
    if matches:
        index = int(matches[0])
        if index >= len(options):
            return select_random_options(options)
        return index
    else:
        return select_random_options(options)
