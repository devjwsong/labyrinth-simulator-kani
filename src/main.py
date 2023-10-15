from socket import timeout
from tracemalloc import start
from utils import select_options, check_init_types
from models.kani_models import generate_engine
from agents.player import Player
from agents.manager import GameManager
from constant_prompts import INSTRUCTION, RULE_SUMMARY, INIT_QUERY
from typing import Dict
from argparse import Namespace
from inputimeout import inputimeout, TimeoutOccurred

import argparse
import json
import logging
import asyncio
import random
import time

log = logging.getLogger("kani")
message_log = logging.getLogger("kani.messages")

PER_PLAYER_TIME = None  # Set to 5 seconds only for action scene.
ONE_HOUR = 60 * 60


def create_character(data: Dict, idx: int=0):
    print(f"<CREATING CHARACTER {idx+1}>")
    print("Before we get started, create your character to play the game.")

    print("In the Labyrinth, there are multiple kins to choose.")
    print("Each kin has unique persona and every character has traits and flaws which might affect your gameplay.")
    print("Each character can select one trait and one flaw, but the number of traits/flaws might vary depending on your kin.")
    while True:
        # Showing the list of kins.
        print()
        print("Select the kin to see more detail.")
        kin = select_options(list(data['kins'].keys()))
        info = data['kins'][kin]
        persona = info['persona']

        # Showing the details of the selected kin.
        print()
        print(f"Kin: {kin}")
        for s, sent in enumerate(persona):
            print(f"{s+1}. {sent}")
        print("-" * 50)
        print(f"{info['guide']}")

        # Confirming the kin.
        print()
        print(f"Are you going to go with {kin}?")
        confirmed = select_options(['yes', 'no'])
        if confirmed == 'no':
            print("Going back to the list...")
            continue

        traits = []
        flaws = []
        items = []

        # Setting the name.
        print()
        print("What is your name?")
        name = input("Input: ")

        # Setting the goal.
        print()
        print("What is your goal? Why did you come to the Labyrinth to challenge the Goblin King?")
        goal = input("Input: ")

        # Setting the character-specific additional features.
        if kin == 'Dwarf':
            print()
            print("You've selected Dwarf. Select your job.")
            jobs_and_tools = info['tables']['jobs_and_tools']
            selected = select_options(jobs_and_tools)
            traits.append(f"My job is {selected['job']}.")

            print()
            print(f"Give more details on your tool: {selected['tool']}")
            item_description = input("Input: ")
            items.append({'name': selected['tool'], 'description': item_description})

        elif kin == 'Firey' or kin == 'Knight of Yore' or kin == 'Worm':
            traits += info['default_traits']

        elif kin == 'Goblin':
            print()
            print(f"You've selected Goblin. Specify why you are against the Goblin King. If you leave an empty string, the default value '{info['default_traits'][0]}' will be added.")
            default_traits = info['default_traits']
            reason = input("Input: ")
            if len(reason) > 0:
                default_traits[0] = reason
            traits += default_traits

        elif kin == 'Horned Beast':
            print()
            print(f"You've selected Horned Beast. Select one object type you can control.")
            object_type = select_options(info['tables']['objects'])
            traits.append(f"I can control an object of type {object_type}.")

            flaws += info['default_flaws']

        # Picking up a trait.
        print()
        print("Now, select one trait from the given list.")
        selected = select_options(data['traits'])
        traits.append(f"{selected['trait']}: {selected['description']}")
        if kin == 'Human':
            extra_traits = [entry for entry in data['traits'] if entry['trait'] != selected['trait']]
            print()
            print(f"You've selected Human. You can pick one more extra trait.")
            selected = select_options(extra_traits)
            traits.append(f"{selected['trait']}: {selected['description']}")

        # Picking up a flaw.
        print()
        print("Next, select one flaw from the given list.")
        filtered_flaws = []
        for entry in data['flaws']:
            included = True
            if 'restriction' in entry:
                for trait in traits:
                    if trait.startswith(entry['restriction']):
                        included = False
                        break
            if included:
                filtered_flaws.append(entry)
        selected = select_options(filtered_flaws)
        flaws.append(f"{selected['flaw']}: {selected['description']}")

        # Finally setting the player instance.
        player = Player(
            name=name,
            kin=kin,
            persona=persona,
            goal=goal,
            traits=traits,
            flaws=flaws,
            items=items
        )
        print()
        print("Finally, confirm if these specifications are matched with your choices.")
        player.show_info()

        print()
        print("Are they correct?")
        confirmed = select_options(['yes', 'no'])
        if confirmed == 'no':
            print("Going back to the list...")
            continue

        print("The player character has been created successfully.")
        return player


def main(manager: GameManager, scene: Dict, args: Namespace):
    print("#" * 100)
    print("--WELCOME TO THE LABYRINTH--")

    # Making player characters.
    print()
    print("CREATE THE PLAYER CHARACTERS.")
    with open("data/characters.json", 'r') as f:
        character_data = json.load(f)
    players = {}
    for p in range(args.num_players):
        print('-' * 100)
        player = create_character(character_data, p)
        players[p+1] = player

    loop = asyncio.get_event_loop()

    # Initializaing the scene.
    print()
    print("INITIALIZING THE SCENE...")
    init_query = '\n'.join([' '. join(query) for query in INIT_QUERY])
    async def scene_init():
        try:
            await manager.init_scene(
                init_query,
                scene,
            )
            check_init_types(manager)
        except:
            log.error("Scene initialization failed. Try again.")
            loop.close()
    loop.run_until_complete(scene_init())

    # Explaining the current scene.
    print()
    print(f"CHAPTER: {manager.chapter}")
    print(f"SCENE: {manager.scene}")
    print(f"{' '.join(manager.scene_summary)}")
    async def main_logic():
        start_time = time.time()
        manager.start_time = start_time

        while True:
            user_queries = []
            for p, player in players.items():
                try:
                    print()
                    query = inputimeout(f"[PLAYER {p} / {player.name.upper()}]: ", timeout=PER_PLAYER_TIME)
                    if len(query) > 0:  # Empty input is ignored.
                        user_queries.append((p, query))
                except TimeoutOccurred:
                    break

            print()
            async for response in manager.full_round(
                user_queries,
                players,
                max_tokens=args.max_tokens,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                temperature=args.temperature,
                top_p=args.top_p
            ):
                print(f"[GOBLIN KING]: {response.content}")

            # Validating the success/failure conditions to terminate the game.
            succ, fail = False, False
            succ = await manager.validate_success_condition()
            fail = await manager.validate_failure_condition()

            if succ and fail:
                print("CONTRADICTORY VALIDATION BETWEEN SUCCESS AND FAILURE. KEEPING THE GAME SCENE MORE.")
            elif succ:
                print("PLAYER WON! ENDING THE CURRENT SCENE.")
                break
            elif fail:
                print("PLAYER LOST! ENDING THE CURRENT SCENE.")
                break

    loop.run_until_complete(main_logic())

    loop.close()

# For debugging.
if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # Arguments for the game play.
    parser.add_argument('--seed', type=int, required=True, help="The random seed for every randomized operation.")
    parser.add_argument('--engine_name', type=str, required=True, help="The engine corresponding the model tested.")
    parser.add_argument('--model_idx', type=str, required=True, help="The model index.")
    parser.add_argument('--rule_injection', type=str, default=None, help="The rule injection type.")
    parser.add_argument('--scene_idx', type=int, help="The index of the scene for the initialization evaluation.")
    parser.add_argument('--num_players', type=int, default=1, help="The number of players.")

    # Parameters for the response generation.
    parser.add_argument('--max_tokens', type=int, default=None, help="The maximum number of tokens to generate.")
    parser.add_argument('--frequency_penalty', type=float, default=0.0, help="A positive value penalizes the repetitive new tokens. (-2.0 - 2.0)")
    parser.add_argument('--presence_penalty', type=float, default=0.0, help="A positive value penalizes the new tokens based on whehter they appear in the text so far. (-2.0 - 2.0)")
    parser.add_argument('--temperature', type=float, default=1.0, help="A higher value makes the output more random. (0.0 - 2.0)")
    parser.add_argument('--top_p', type=float, default=1.0, help="The probability mass which will be considered for the nucleus sampling. (0.0 - 1.0)")

    args = parser.parse_args()

    assert args.rule_injection in [None, 'full', 'retrieval'], "Either specify an available rule injection option: 'full' / 'retrieval', or leave it as None."

    # Creating the engine.
    random.seed(args.seed)
    engine = generate_engine(engine_name=args.engine_name, model_idx=args.model_idx)

    # Setting the system prompt.
    system_prompt = ' '.join(INSTRUCTION)
    if args.rule_injection == 'full':
        rule_summary = '\n'.join([' '. join(rule) for rule in RULE_SUMMARY])
        system_prompt = f"{system_prompt}\nHere are the rules of the Labyrinth you should follow.\n{rule_summary}"
    elif args.rule_injection == 'retrieval':
        # TODO: Adding after the RAG method is completed.
        pass

    # Initializing the game manager.
    manager = GameManager(engine=engine, system_prompt=system_prompt)

    # Loading the scene file.
    with open("data/scenes.json", 'r') as f:
        scenes = json.load(f)

    assert args.scene_idx is not None, "The scene index should be provided."
    assert 0 <= args.scene_idx < len(scenes), "The scene index is not valid."

    main(manager, scenes[args.scene_idx], args)

