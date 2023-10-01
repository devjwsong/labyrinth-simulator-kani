from agents.manager import GameManager
from models.kani_models import generate_engine
from utils import select_options
from constant_prompts import INSTRUCTION, RULE_SUMMARY, INIT_QUERY
from typing import Dict

import asyncio
import argparse
import json


def evaluate_init(manager: GameManager, scene: Dict):
    # Converting the query into string.
    init_query = '\n'.join([' '. join(query) for query in INIT_QUERY])

    async def test():
        await manager.init_scene(
            init_query,
            scene,
        )
    asyncio.run(test())

    # TODO: How to export the export results?
    print("#" * 100)
    manager.show_scene()


def evaluate_rules(manager: GameManager):
    # The list of test questions.
    questions = [
        "What attributes does each player character have?"
    ]

    # The list of the user scores.
    options = [
        {'score': 1.0, 'description': "Perfectly correct."},
        {'score': 0.5, 'description': "Partially correct. (e.g. dropping essential information, faking up the false rules...)"},
        {'score': 0.0, 'description': "Completely wrong."}
    ]
    scores = []

    async def test():
        for q, question in enumerate(questions):
            query = f"Answer the following question according to the Labyrinth's rules.\n{question}"
            response = await manager.chat_round_str(query)
            print()
            print(f"QUESTION {q+1}: {question}")
            print(f"ANSWER: {response}")

            # Recording the user score.
            print("Select the score for the given response.")
            selected = select_options(options)
            scores.append(selected['score'])

            # Clearing the chat history.
            manager.chat_history = []
    asyncio.run(test())

    # TODO: How to export the export results?
    print("#" * 100)
    print(scores)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_name', type=str, required=True, help="The evaluation name.")
    parser.add_argument('--engine_name', type=str, required=True, help="The engine corresponding the model tested.")
    parser.add_argument('--model_idx', type=str, required=True, help="The model index.")
    parser.add_argument('--rule_injection', type=str, required=False, help="The rule injection type.")
    parser.add_argument('--scene_idx', type=int, help="The index of the scene for the initialization evaluation.")

    args = parser.parse_args()

    assert args.eval_name in ['init', 'rules'], "Specify the correct evaluation name."
    assert args.rule_injection in [None, 'full', 'retrieval'], "Either specify an available rule injection option: 'full' / 'retrieval', or leave it as None."

    # Creating the engine.
    engine = generate_engine(engine_name=args.engine_name, model_idx=args.model_idx)

    # Setting the system prompt.
    system_prompt = ' '.join(INSTRUCTION)
    if args.rule_injection == 'full':
        rule_summary = '\n'.join([' '. join(rule) for rule in RULE_SUMMARY])
        system_prompt = f"{system_prompt}Here are the rules of the Labyrinth you should follow.\n{rule_summary}"
    elif args.rule_injection == 'retrieval':
        # TODO: Adding after the RAG method is completed.
        pass

    # Initializing the game manager.
    manager = GameManager(engine=engine, system_prompt=system_prompt)

    if args.eval_name == 'init':
        # Loading the scene file.
        with open("data/scenes.json", 'r') as f:
            scenes = json.load(f)

        assert args.scene_idx is not None, "The scene index should be provided."
        assert 0 <= args.scene_idx < len(scenes), "The scene index is not valid."

        evaluate_init(manager, scenes[args.scene_idx])
    elif args.eval_name == 'rules':
        evaluate_rules(manager)
