from utils import select_options, check_init_types, print_logic_start, print_question_start, print_system_log, print_manager_log, get_player_input, logic_break
from agents.kani_models import generate_engine
from kani.utils.message_formatters import assistant_message_contents_thinking
from agents.player import Player
from agents.manager import GameManager
from sentence_transformers import SentenceTransformer
from constants import INSTRUCTION, RULE_SUMMARY, INIT_QUERY, TOTAL_TIME, PER_PLAYER_TIME, ONE_HOUR
from typing import Dict
from argparse import Namespace
from inputimeout import TimeoutOccurred

import argparse
import json
import logging
import asyncio
import random
import time
import torch

log = logging.getLogger("kani")
message_log = logging.getLogger("kani.messages")


def create_character(data: Dict):
    print_system_log("BEFORE WE GET STARTED, CREATE YOUR CHARACTER TO PLAY THE GAME.")

    print_system_log("IN THE LABYRINTH, THERE ARE MULTIPLE KINS TO CHOOSE.")
    print_system_log("EACH KIN HAS UNIQUE PERSONA AND EVERY CHARACTER HAS TRAITS AND FLAWS WHICH MIGHT AFFECT YOUR GAMEPLAY.")
    print_system_log("EACH CHARACTER CAN SELECT ONE TRAIT AND ONE FLAW, BUT THE NUMBER OF TRAITS/FLAWS MIGHT VARY DEPENDING ON YOUR KIN.", after_break=True)
    while True:
        # Showing the list of kins.
        print_system_log("SELECT THE KIN TO SEE MORE DETAIL.")
        kins = list(data['kins'].keys())
        selected = select_options(kins)
        kin = kins[selected]
        info = data['kins'][kin]
        persona = info['persona']

        # Showing the details of the selected kin.
        print_system_log("INFORMATION ON THE SELECTED KIN IS...")
        print(f"Kin: {kin}")
        for s, sent in enumerate(persona):
            print(f"({s+1}) {sent}")
        print(' '.join(info['guide']))

        # Confirming the kin.
        print_question_start()
        print_system_log(f"ARE YOU GOING TO GO WITH {kin}?")
        selected = select_options(['Yes', 'No'])
        if selected == 1:
            print_system_log("GOING BACK TO THE LIST...")
            continue

        traits = {}
        flaws = {}
        inventory = {}

        # Setting the name.
        print_question_start()
        print_system_log("WHAT IS YOUR NAME?")
        name = get_player_input(after_break=True)

        # Setting the goal.
        print_question_start()
        print_system_log("WHAT IS YOUR GOAL? WHY DID YOU COME TO THE LABYRINTH TO CHALLENGE THE GOBLIN KING?")
        goal = get_player_input(after_break=True)

        # Setting the character-specific additional features.
        if kin == 'Dwarf':
            print_question_start()
            print_system_log("YOU'VE SELECTED DWARF. SELECT YOUR JOB.")
            jobs_and_tools = info['tables']['jobs_and_tools']
            selected = select_options(jobs_and_tools)
            job_and_tool = jobs_and_tools[selected]
            traits['Job'] = f"My job is {job_and_tool['job']} and I can use my {job_and_tool['tool']} professionally."

            print_question_start()
            print_system_log(f"GIVE MORE DETAILS ON YOUR TOOL: {job_and_tool['tool']}")
            item_description = get_player_input(after_break=True)
            inventory[job_and_tool['tool']] = item_description

        elif kin == 'Firey' or kin == 'Knight of Yore' or kin == 'Worm':
            traits.update(info['default_traits'])

        elif kin == 'Goblin':
            print_question_start()
            print_system_log("YOU'VE SELECTED GOBLIN. SPECIFY WHY YOU ARE AGAINST THE GOBLIN KING.")
            default_traits = info['default_traits']
            reason = get_player_input(after_break=True)
            if len(reason) > 0:
                default_traits['Goblin feature'] = reason
            traits.update(default_traits)

        elif kin == 'Horned Beast':
            print_question_start()
            print_system_log("YOU'VE SELECTED HORNED BEAST. SELECT ONE OBJECT TYPE YOU CAN CONTROL.")
            objects = info['tables']['objects']
            selected = select_options(objects)
            traits['Control object'] = f"I can control an object of type {objects[selected]}."
            flaws.update(info['default_flaws'])

        # Picking up a trait.
        print_question_start()
        print_system_log("NOW, SELECT ONE TRAIT FROM THE GIVEN LIST.")
        cands = data['traits']
        selected = select_options(cands)
        traits[cands[selected]['trait']] = cands[selected]['description']
        if kin == 'Human':
            extra_cands = [entry for entry in data['traits'] if entry['trait'] not in traits]
            print_question_start()
            print_system_log("YOU'VE SELECTED HUMAN. YOU CAN PICK ONE MORE EXTRA TRAIT.")
            selected = select_options(extra_cands)
            traits[extra_cands[selected]['trait']] = extra_cands[selected]['description']
            
        # Picking up a flaw.
        print_question_start()
        print_system_log("NEXT, SELECT ONE FLAW FROM THE GIVEN LIST.")
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
        flaws[filtered_flaws[selected]['flaw']] = filtered_flaws[selected]['description']

        # Finally setting the player instance.
        player = Player(
            name=name,
            kin=kin,
            persona=persona,
            goal=goal,
            traits=traits,
            flaws=flaws,
            inventory=inventory
        )
        print_question_start()
        print_system_log("FINALLY, CONFIRM IF THESE SPECIFICATIONS ARE MATCHED WITH YOUR CHOICES.")
        player.show_info()

        print_question_start()
        print_system_log("ARE THEY CORRECT?")
        selected = select_options(['Yes', 'No'])
        if selected == 1:
            print_system_log("GOING BACK TO THE LIST...", after_break=True)
            continue

        print_system_log("THE PLAYER CHARACTER HAS BEEN CREATED SUCCESSFULLY.")
        return player


def main(manager: GameManager, scene: Dict, args: Namespace):
    # Making player characters.
    with open("data/characters.json", 'r') as f:
        character_data = json.load(f)
    players = {}
    for p in range(args.num_players):
        print_logic_start(f"CHARACTER {p+1} CREATION")
        player = create_character(character_data)
        players[p+1] = player
        logic_break()
    manager.players = players
    manager.name_to_idx = {player.name: idx for idx, player in players.items()}

    loop = asyncio.get_event_loop()

    # Initializaing the scene.
    print_system_log("INITIALIZING THE SCENE...")
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

    # DEBUG
    manager.show_scene()

    # Explaining the current scene.
    print_logic_start("GAME START.")
    print_system_log(f"CHAPTER: {manager.chapter}")
    print_system_log(f"SCENE: {manager.scene}")
    print_system_log(f"{' '.join(manager.scene_summary)}", after_break=True)
    async def main_logic():
        start_time = time.time()
        notified = 0

        while True:
            # Checking if this is an action scene now.
            per_player_time = PER_PLAYER_TIME if manager.is_action_scene else None

            # Calculating the elapsed time.
            elapsed_time = int(time.time() - start_time)
            if elapsed_time >= (notified * ONE_HOUR):
                hours, minutes, seconds = elapsed_time // 3600, (elapsed_time % 3600) // 60, elapsed_time % 60
                print_system_log(f"{hours} hours {minutes} minutes {seconds} seconds have passed from the start of the game.", after_break=True)
                notified += 1

            user_queries = []
            for p, player in players.items():
                try:
                    query = get_player_input(name=player.name, per_player_time=per_player_time, after_break=True)
                    if len(query) > 0:  # Empty input is ignored.
                        user_queries.append((p, query))
                except TimeoutOccurred:
                    break

            async for response in manager.full_round_str(
                user_queries,
                message_formatter=assistant_message_contents_thinking,
                max_tokens=args.max_tokens,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                temperature=args.temperature,
                top_p=args.top_p
            ):
                print_manager_log(response, after_break=True)

            # Validating the success/failure conditions to terminate the game.
            succ, fail = False, False
            succ = await manager.validate_success_condition()
            fail = await manager.validate_failure_condition()
            if elapsed_time >= TOTAL_TIME:
                print("PLAYER LOST! ENDING THE CURRENT SCENE.")
                print("[TIME LIMIT REACHED.]")
                break

            if succ and fail:
                print("CONTRADICTORY VALIDATION BETWEEN SUCCESS AND FAILURE. KEEPING THE GAME SCENE MORE.")
            elif succ:
                print("PLAYER WON! ENDING THE CURRENT SCENE.")
                print(f"[SUCCESS CONDITION] {manager.success_condition}")
                break
            elif fail:
                print("PLAYER LOST! ENDING THE CURRENT SCENE.")
                print(f"[FAILURE CONDITION] {manager.failure_condition}")
                break
        logic_break()

    loop.run_until_complete(main_logic())
    loop.close()

# For debugging.
if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # Arguments for the gameplay.
    parser.add_argument('--seed', type=int, required=True, help="The random seed for randomized operations.")
    parser.add_argument('--engine_name', type=str, required=True, help="The name of the engine for running kani corresponding the language model used.")
    parser.add_argument('--model_idx', type=str, required=True, help="The index of the model.")
    parser.add_argument('--rule_injection', type=str, default=None, help="The rule injection policy.")
    parser.add_argument('--scene_idx', type=int, help="The index of the scene to play.")
    parser.add_argument('--num_players', type=int, default=1, help="The number of players.")

    # Parameters for the prompt construction.
    parser.add_argument('--concat_policy', type=str, default='simple', help="The concatenation policy for including the previous chat logs.")
    parser.add_argument('--max_turns', type=int, default=None, help="The maximum number of turns to be included.")
    parser.add_argument('--summarization', action='store_true', help="Setting whether to include the summarization or not.")
    parser.add_argument('--summ_period', type=int, default=None, help="The summarization period in terms of the number of turns.")
    parser.add_argument('--clear_raw_logs', action='store_true', help="Setting whether to remove the raw chat logs after the summarization.")

    # Parameters for the response generation.
    parser.add_argument('--max_tokens', type=int, default=None, help="The maximum number of tokens to generate.")
    parser.add_argument('--frequency_penalty', type=float, default=0.0, help="A positive value penalizes the repetitive new tokens. (-2.0 - 2.0)")
    parser.add_argument('--presence_penalty', type=float, default=0.0, help="A positive value penalizes the new tokens based on whether they appear in the text so far. (-2.0 - 2.0)")
    parser.add_argument('--temperature', type=float, default=1.0, help="A higher value makes the output more random. (0.0 - 2.0)")
    parser.add_argument('--top_p', type=float, default=1.0, help="The probability mass which will be considered for the nucleus sampling. (0.0 - 1.0)")

    args = parser.parse_args()

    assert args.rule_injection in [None, 'full', 'retrieval'], "Either specify an available rule injection option: 'full' / 'retrieval', or leave it as non-specified."
    if not args.summarization:
        assert args.summ_period is None, "To use summ_period, you must set the summarization argument."
        assert args.clear_raw_logs is False, "To use clear_raw_logs, you must set the summarization argument."
    if args.summarization and args.summ_period is None:
        print_system_log("SUMMARIZATION WITHOUT PERIOD WILL IGNORE ALL OTHER SETTINGS FOR PROMPT. THE WHOLE CHAT LOGS WILL BE SUMMARIZED INTO A PROMPT.")
    else:
        assert args.concat_policy in ['simple', 'retrieval'], "The concatenation policy should be either 'simple' or 'retrieval'."
        if args.max_turns is None:
            print_system_log("ANY CONCATENATION POLICY WITH NO SPECIFIC MAX NUMBER OF TURNS WOULD BE CASTED INTO THE SIMPLE CONCATENATION.")
            args.concat_policy = 'simple'  # The retrieval concatenation without any number of turns is not different from the simple concatenation.

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

    # Intializing the sentence encoder if the concatenation policy is retrieval.
    encoder = None
    if args.concat_policy == 'retrieval':
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        encoder = SentenceTransformer('all-mpnet-base-v2').to(device)

    # Initializing the game manager.
    manager = GameManager(
        main_args=args,
        encoder=encoder,
        engine=engine, 
        system_prompt=system_prompt
    )

    # Loading the scene file.
    with open("data/scenes.json", 'r') as f:
        scenes = json.load(f)

    assert args.scene_idx is not None, "The scene index should be provided."
    assert 0 <= args.scene_idx < len(scenes), "The scene index is not valid."

    main(manager, scenes[args.scene_idx], args)

