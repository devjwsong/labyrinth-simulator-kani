from agents.manager import GameManager
from models.kani_models import generate_engine
from constant_prompts import INSTRUCTION, RULE_SUMMARY, INIT_QUERY

import asyncio
import argparse
import json


def evaluate_init(engine, scene, rule_injection=None):
    # Setting the system prompt.
    system_prompt = ' '.join(INSTRUCTION)
    if rule_injection == 'full':
        rule_summary = '\n'.join([' '. join(rule) for rule in RULE_SUMMARY])
        system_prompt = f"{system_prompt}Here are the rules of the Labyrinth you should follow.\n{rule_summary}"
    elif rule_injection == 'retrieval':
        # TODO: Adding after the RAG method is completed.
        pass

    # Initializing the game manager.
    manager = GameManager(engine=engine, system_prompt=system_prompt)

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

    if args.eval_name == 'init':
        # Loading the scene file.
        with open("data/scenes.json", 'r') as f:
            scenes = json.load(f)

        assert args.scene_idx is not None, "The scene index should be provided."
        assert 0 <= args.scene_idx < len(scenes), "The scene index is not valid."

        evaluate_init(engine, scenes[args.scene_idx], args.rule_injection)
