from utils import log_break, select_options, print_logic_start, print_question_start, print_system_log, print_player_log, print_manager_log, get_player_input, logic_break
from kani.utils.message_formatters import assistant_message_contents_thinking
from kani.models import ChatMessage
from kani.engines.openai import OpenAIEngine
from agents.player import Player, PlayerKani
from agents.manager import GameManager
from constants import ASSISTANT_INSTRUCTION, USER_INSTRUCTION, GAME_TIME_LIMIT, SYSTEM_TIME_LIMIT,  PER_PLAYER_TIME, ONE_HOUR
from typing import Dict
from argparse import Namespace
from inputimeout import TimeoutOccurred
from datetime import datetime
from pytz import timezone

import argparse
import json
import logging
import asyncio
import random
import time
import os

log = logging.getLogger("kani")
message_log = logging.getLogger("kani.messages")


# Loading a player character which was created before.
def load_player_character(data: Dict, engine: OpenAIEngine, automated_player: bool):
    if automated_player:
        system_prompt = ' '.join(USER_INSTRUCTION)
        player = PlayerKani(
            engine=engine,
            system_prompt=system_prompt,
            name=data['name'],
            kin=data['kin'],
            persona=data['persona'],
            goal=data['goal'],
            traits=data['traits'],
            flaws=data['flaws'],
            inventory=data['inventory'],
            additional_notes=data['additional_notes']
        )
    else:
        player = Player(
            name=data['name'],
            kin=data['kin'],
            persona=data['persona'],
            goal=data['goal'],
            traits=data['traits'],
            flaws=data['flaws'],
            inventory=data['inventory'],
            additional_notes=data['additional_notes']
        )

    return player


def main(manager: GameManager, args: Namespace):
    loop = asyncio.get_event_loop()

    # Explaining the current scene.
    start_sent = "GAME START."
    print_logic_start(start_sent)
    scene_intro = f"\nCHAPTER: {manager.chapter}\nSCENE: {manager.scene}\n{' '.join(manager.scene_summary)}"
    print_system_log(scene_intro, after_break=True)
    async def game_logic():
        start_time = time.time()
        notified = 0

        player_queries = [[ChatMessage.system(content=f"{start_sent}{scene_intro}")] for _ in range(len(manager.players))]
        manager_queries = []
        while True:
            # Checking if this is an action scene now.
            per_player_time = PER_PLAYER_TIME if manager.is_action_scene else None

            # Calculating the elapsed time.
            elapsed_time = int(time.time() - start_time)
            if elapsed_time >= (notified * ONE_HOUR):
                hours, minutes, seconds = elapsed_time // 3600, (elapsed_time % 3600) // 60, elapsed_time % 60
                print_system_log(f"{hours} hours {minutes} minutes {seconds} seconds have passed from the start of the game.", after_break=True)
                notified += 1

            # Random shuffling the order of players every turn.
            player_idxs = list(range(len(manager.players)))
            random.shuffle(player_idxs)

            for p in player_idxs:
                player = players[p]
                try:
                    if isinstance(player, PlayerKani):
                        player_query = await player.chat_round_str(player_queries[p])
                        print_player_log(player_query, player.name, after_break=True)

                    else:
                        player_query = get_player_input(name=player.name, per_player_time=per_player_time, after_break=True)
                        if len(player_query) > 0:  # Empty input is ignored.
                            if player_query == "Abort!":  # Immediate termination.
                                print_system_log("THE GAME WAS ABORTED BY THE USER REQUEST.")
                                manager.gameplay_logs.append({
                                    'game_result': 'aborted',
                                    'condition': "The user intentionally stopped the game."
                                })
                                return

                    for pp in player_idxs:
                        if pp != p: 
                            player_queries[pp].append(ChatMessage.user(name=player.name, content=player_query))
                        else:
                            player_queries[pp] = []
                    manager_queries.append(ChatMessage.user(name=player.name, content=player_query))

                except TimeoutOccurred:
                    continue
            
            async for response in manager.full_round_str(
                manager_queries,
                message_formatter=assistant_message_contents_thinking,
                include_functions=args.include_functions,
                max_tokens=args.max_tokens,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                temperature=args.temperature,
                top_p=args.top_p
            ):
                for p in range(len(manager.players)):
                    player_queries[p].append(ChatMessage.user(name="Goblin_King", content=response))
                manager_queries = []
                print_manager_log(response, after_break=True)

            # Validating the success/failure conditions to terminate the game.
            succ = await manager.validate_success_condition()
            fail = await manager.validate_failure_condition()
            if elapsed_time >= GAME_TIME_LIMIT:
                print_system_log("PLAYER LOST! ENDING THE CURRENT SCENE.")
                print_system_log("TIME LIMIT REACHED.")
                manager.gameplay_logs.append({
                    'game_result': 'timeout',
                    'condition': 'The players failed to beat the game in the time limit.'
                })
                break

            if succ and fail:
                print_system_log("CONTRADICTORY VALIDATION BETWEEN SUCCESS AND FAILURE. KEEPING THE GAME SCENE MORE.")
            elif succ:
                print_system_log("PLAYER WON! ENDING THE CURRENT SCENE.")
                print_system_log(f"SUCCESS CONDITION: {manager.success_condition}")
                manager.gameplay_logs.append({
                    'game_result': 'success',
                    'condition': manager.success_condition
                })
                break
            elif fail:
                print_system_log("PLAYER LOST! ENDING THE CURRENT SCENE.")
                print_system_log(f"FAILURE CONDITION: {manager.failure_condition}")
                manager.gameplay_logs.append({
                    'game_result': 'failure',
                    'condition': manager.failure_condition
                })
                break

        logic_break()

    async def main_logic():
        try:
            await asyncio.wait_for(game_logic(), SYSTEM_TIME_LIMIT)
        except asyncio.TimeoutError:
            print_system_log("THE GAME GOT STUCK DUE TO UNKNOWN TECHNICAL REASON.")
            manager.gameplay_logs.append({
                'game_result': 'unexpected',
                'condition': "The game stuck due to an unexpected behavior."
            })

    loop.run_until_complete(main_logic())
    loop.close()

# For debugging.
if __name__=='__main__':
    print_question_start()
    print_system_log("BEFORE STARTING THE GAME, GIVE US YOUR USERNAME, WHICH IS USED FOR RECORDING PURPOSE.")
    owner_name = get_player_input(after_break=True)

    now = datetime.now(timezone('US/Eastern'))
    execution_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    parser = argparse.ArgumentParser()

    # Arguments for the gameplay.
    parser.add_argument('--seed', type=int, required=True, help="The random seed for randomized operations.")
    parser.add_argument('--model_idx', type=str, required=True, help="The index of the model.")
    parser.add_argument('--rule_injection', type=str, default='full', help="The rule injection policy.")
    parser.add_argument('--scene_path', type=str, required=True, help="The path of the JSON file which has the initialized scene information before.")
    parser.add_argument('--players_path', type=str, required=True, help="The path of the JSON file which has the created player character information before.")
    parser.add_argument('--export_data', action='store_true', help="Setting whether to export the gameplay data after the game for the evaluation purpose.")
    parser.add_argument('--num_ai_players', type=int, default=0, help="The number of AI players to simulate.")

    # Parameters for the prompt construction.
    parser.add_argument('--concat_policy', type=str, default='simple', help="The concatenation policy for including the previous chat logs.")
    parser.add_argument('--max_num_msgs', type=int, help="The maximum number of messages to be included in the prompt as chat history.")
    parser.add_argument('--summarization', action='store_true', help="Setting whether to include the summarization or not.")
    parser.add_argument('--summ_period', type=int, help="The summarization period in terms of the number of turns.")
    parser.add_argument('--clear_raw_logs', action='store_true', help="Setting whether to remove the raw chat logs after the summarization.")

    # Parameters for togging the additional contexts.
    parser.add_argument('--include_functions', action='store_true', help="Setting whether to use function calls or not.")
    parser.add_argument('--include_rules', action='store_true', help="Setting whether to include the game rules in the prompt.")
    parser.add_argument('--include_scene_state', action='store_true', help="Setting whether to include the state of the current scene.")
    parser.add_argument('--include_player_states', action='store_true', help="Setting whether to include the states of the players.")
    parser.add_argument('--update_states', action='store_true', help="Setting whether to use a model to directly updates the scene/player states.")

    # Parameters for the response generation.
    parser.add_argument('--max_tokens', type=int, help="The maximum number of tokens to generate.")
    parser.add_argument('--frequency_penalty', type=float, default=0.5, help="A positive value penalizes the repetitive new tokens. (-2.0 - 2.0)")
    parser.add_argument('--presence_penalty', type=float, default=0.5, help="A positive value penalizes the new tokens based on whether they appear in the text so far. (-2.0 - 2.0)")
    parser.add_argument('--temperature', type=float, default=1.0, help="A higher value makes the output more random. (0.0 - 2.0)")
    parser.add_argument('--top_p', type=float, default=0.8, help="The probability mass which will be considered for the nucleus sampling. (0.0 - 1.0)")

    args = parser.parse_args()

    assert args.rule_injection in ['full', 'retrieval'], "Specify an available rule injection option: 'full' / 'retrieval', or leave it as non-specified."
    if not args.summarization:
        assert args.summ_period is None, "To use summ_period, you must set the summarization argument."
        assert args.clear_raw_logs is False, "To use clear_raw_logs, you must set the summarization argument."
    if args.summarization and args.summ_period is None:
        print_system_log("SUMMARIZATION WITHOUT PERIOD WILL IGNORE ALL OTHER SETTINGS FOR PROMPT. THE WHOLE CHAT LOGS WILL BE SUMMARIZED INTO A PROMPT.")
    else:
        assert args.concat_policy in ['simple', 'retrieval'], "The concatenation policy should be either 'simple' or 'retrieval'."
        if args.max_num_msgs is None:
            print_system_log("ANY CONCATENATION POLICY WITH NO SPECIFIC MAX NUMBER OF MESSAGES WOULD BE CASTED INTO THE SIMPLE CONCATENATION.")
            args.concat_policy = 'simple'  # The retrieval concatenation without any number of turns is not different from the simple concatenation.
    if args.update_states:
        print_system_log("YOU SET update_state=True WHICH AUTOMATICALLY TURNS OFF include_scene_state AND include_player_states.")
        args.include_scene_state = False
        args.include_player_states = False

    # Creating the engine.
    random.seed(args.seed)
    api_key = input("Enter the API key for OpenAI API: ")
    log_break()
    engine = OpenAIEngine(api_key, model=args.model_idx)

    # Initializing the game manager.
    print_system_log("LOADING THE SCENE...")
    with open(args.scene_path, 'r') as f:
        scene = json.load(f)

    system_prompt = ' '.join(ASSISTANT_INSTRUCTION)
    manager = GameManager(
        scene=scene,
        main_args=args,
        engine=engine, 
        system_prompt=system_prompt
    )

    # DEBUG
    manager.show_scene()
    log_break()

    # Setting the players.
    print_system_log("LOADING THE CREATED PLAYER INFORMATION...")
    with open(args.players_path, 'r') as f:
        player_data = json.load(f)
    assert args.num_ai_players <= len(player_data), f"The number of AI players cannot exceed the total number of players: {len(player_data)}."
    
    # Iterating the player character instantiation.
    players = []
    num_left = len(player_data) - args.num_ai_players
    for p, data in enumerate(player_data):
        print_logic_start(f"CHARACTER {p+1} INFORMATION")
        print(data)
        
        automated_player = True
        if num_left > 0:
            print_system_log(f"WOULD YOU LIKE TO PLAY AS THIS CHARACTER? (THE NUMBER OF AVAILABLE HUMAN PLAYERS: {num_left})")
            selected = select_options(['Yes', 'No'])
            if selected == 0:
                automated_player = False
                num_left -= 1
        else:
            print_system_log(f"THE AVAILABLE NUMBER OF HUMAN PLAYERS IS 0. INITIALIZING THIS CHARACTER INTO AN AI...")

        player = load_player_character(data, manager.engine, automated_player)
        players.append(player)
        logic_break()
    manager.players = players
    manager.name_to_idx = {player.name: idx for idx, player in enumerate(players)}

    # The main game logic.
    main(manager, args)

    # Exporting data after finishing the scene.
    if args.export_data:
        file_dir = f"results/{args.scene_path.split('/')[1]}/rule_injection={args.rule_injection}/concat={args.concat_policy}/" + \
            f"msg_limit={args.max_num_msgs}/summarization={args.summarization}/summ_period={args.summ_period}/clear_raw={args.clear_raw_logs}/" + \
            f"functions={args.include_functions}/rules={args.include_rules}/scene_state={args.include_scene_state}/player_states={args.include_player_states}/" + \
            f"update_states={args.update_states}"
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)

        file_path = f"{file_dir}/{owner_name}-model={args.model_idx}-seed={args.seed}-time={execution_time}.json"
        with open(file_path, 'w') as f:
            json.dump(manager.gameplay_logs, f)
