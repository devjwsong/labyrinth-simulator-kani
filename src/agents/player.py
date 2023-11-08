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
