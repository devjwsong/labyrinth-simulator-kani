from kani import Kani
from argparse import Namespace
from utils import print_system_log


# Default Player class.
class Player():
    def __init__(self, main_args: Namespace):
        # Player character info based on the sheet in the text book.
        self.name = main_args['name']
        self.kin = main_args['kin']
        self.persona = main_args['persona']
        self.goal = main_args['goal']

        self.traits = main_args['traits']
        self.flaws = main_args['flaws']
        self.inventory = main_args['inventory']

    # Getter for persona in a (numbered) list format.
    def get_persona(self, with_number=False):
        if with_number:
            return [f"({s+1}) {sent}" for s, sent in enumerate(self.persona)]
        return self.persona

    # Getter for traits in a (numbered) list format.
    def get_traits(self, with_number=False):
        if with_number:
            return [f"({t+1}) {trait} - {desc}" for t, (trait, desc) in enumerate(self.traits.items())]
        return [f"{trait} - {desc}" for trait, desc in self.traits.items()]

    # Getter for flaws in a (numbered) list format.
    def get_flaws(self, with_number=False):
        if with_number:
            return [f"({f+1}) {flaw} - {desc}" for f, (flaw, desc) in enumerate(self.flaws.items())]
        return [f"{flaw} - {desc}" for flaw, desc in self.flaws.items()]

    # Getter for inventory in a (numbered) list format.
    def get_inventory(self, with_number=False):
        if with_number:
            return [f"({i+1}) {item} - {desc}" for i, (item, desc) in enumerate(self.inventory.items())]
        return [f"{item} - {desc}" for item, desc in self.inventory.items()]

    # Printing the character sheet so far.
    def show_info(self):
        print(f"NAME: {self.name}")
        print(f"KIN: {self.kin}")
        
        print("PERSONA")
        print('\n'.join(self.get_persona(with_number=True)))

        print(f"GOAL: {self.goal}")

        print("TRAITS")
        print('\n'.join(self.get_traits(with_number=True)))

        print("FLAWS")
        print('\n'.join(self.get_flaws(with_number=True)))

        print("INVENTORY")
        print('\n'.join(self.get_inventory(with_number=True)))
            
    # Adding a trait.
    def add_trait(self, trait, desc):
        self.traits[trait] = desc
        
        # Updating the new chat message.
        msg = f"PLAYER {self.name.upper()} ADDED A TRAIT '{trait}: {desc}'"
        print_system_log(msg, after_break=True)
        return msg
    
    # Adding a flaw.
    def add_flaw(self, flaw, desc):
        self.flaws[flaw] = desc

        # Updating the new chat message.
        msg = f"PLAYER {self.name.upper()} ADDED A FLAW '{flaw}: {desc}'"
        print_system_log(msg, after_break=True)
        return msg
    
    # Adding an item.
    def add_item(self, item, desc):
        self.inventory[item] = desc

        # Updating the new chat message.
        msg = f"PLAYER {self.name.upper()} ADDED AN ITEM '{item}: {desc}' IN THE INVENTORY."
        print_system_log(msg, after_break=True)
        return msg

    # Removing a trait.
    def remove_trait(self, trait):
        self.traits.pop(trait)

        # Updating the new chat message.
        msg = f"PLAYER {self.name.upper()} REMOVED THE TRAIT '{trait}'."
        print_system_log(msg, after_break=True)
        return msg

    # Removing a flaw.
    def remove_flaw(self, flaw):
        self.flaws.pop(flaw)

        # Updating the new chat message.
        msg = f"PLAYER {self.name.upper()} REMOVED THE FLAW '{flaw}'."
        print_system_log(msg, after_break=True)
        return msg

    # Removing an item.
    def remove_item(self, item):
        self.inventory.pop(item)

        # Updating the new chat message.
        msg = f"PLAYER {self.name.upper()} REMOVED THE ITEM '{item}'."
        print_system_log(msg, after_break=True)
        return msg


# Kani version of Player class.
class PlayerKani(Player, Kani):
    def __init__(self, main_args: Namespace, *args, **kwargs):
        Player.__init__(main_args)
        Kani.__init__(*args, **kwargs)
