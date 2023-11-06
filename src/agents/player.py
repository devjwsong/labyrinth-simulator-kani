# The player class.
import enum


class Player():
    def __init__(self, **kwargs):
        # Player character info based on the sheet in the text book.
        self.name = kwargs['name']
        self.kin = kwargs['kin']
        self.persona = kwargs['persona']
        self.goal = kwargs['goal']

        self.traits = kwargs['traits']
        self.flaws = kwargs['flaws']
        self.inventory = kwargs['inventory']

    # Getter for traits with the natural format.
    def get_traits(self):
        return [f"{trait}: {desc}" for trait, desc in self.traits.items()]

    # Getter for flaws with the natural format.
    def get_flaws(self):
        return [f"{flaw}: {desc}" for flaw, desc in self.flaws.items()]

    # Getter for inventory with the natural format.
    def get_inventory(self):
        return [f"{name}: {desc}" for name, desc in self.inventory.items()]

    # Printing the character sheet so far.
    def show_info(self):
        print(f"NAME: {self.name}")
        print(f"KIN: {self.kin}")
        
        print("PERSONA")
        for l, line in enumerate(self.persona):
            print(f"({l+1}) {line}")

        print(f"GOAL: {self.goal}")

        print("TRAITS")
        for l, line in enumerate(self.get_traits()):
            print(f"({l+1}) {line}")

        print("FLAWS")
        for l, line in enumerate(self.get_flaws()):
            print(f"({l+1}) {line}")

        print("INVENTORY")
        for l, line in enumerate(self.get_inventory()):
            print(f"({l+1}) {line}")
            
    # Adding trait.
    def add_trait(self, trait, desc):
        self.traits[trait] = desc
    
    # Adding flaw.
    def add_flaw(self, flaw, desc):
        self.flaws[flaw] = desc
    
    # Adding item.
    def add_item(self, name, desc):
        self.inventory[name] = desc

    # Removing trait.
    def remove_trait(self, trait):
        self.traits.pop(trait)

    # Removing flaw.
    def remove_flaw(self, flaw):
        self.flaws.pop(flaw)

    # Removing item.
    def remove_item(self, name):
        self.inventory.pop(name)
