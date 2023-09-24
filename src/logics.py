from typing import List, Any
from agents.player import Player

import json


def select_options(options: List[Any]):
    while True:
        for o, option in enumerate(options):
            print(f"{o+1}. {option}")
        res = input("Input: ")

        try:
            res = int(res)
            if res < 1 or res > len(options):
                print(f"The allowed value is from {1} to {len(options)}.")
            else:
                return options[res-1]
        except ValueError:
            print("The input should be an integer.")


def create_character():
    with open("data/characters.json", 'r') as f:
        data = json.load(f)

    print("<CREATING CHARACTERS>")
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
