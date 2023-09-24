# The player class.
class Player():
    def __init__(self, **kwargs):
        # Player character info based on the sheet in the text book.
        self.name = kwargs['name']
        self.kin = kwargs['kin']
        self.persona = kwargs['persona']
        self.goal = kwargs['goal']

        self.traits = kwargs['traits']
        self.flaws = kwargs['flaws']
        self.items = kwargs['items']

    # Printing the character sheet so far.
    def show_info(self):
        print("#" * 50)
        print(f"NAME: {self.name}")
        print(f"KIN: {self.kin}")
        
        print("PERSONA")
        for s, sent in enumerate(self.persona):
            print(f"{s+1}. {sent}")

        print(f"GOAL: {self.goal}")

        print("TRAITS")
        for s, sent in enumerate(self.traits):
            print(f"{s+1}. {sent}")

        print("FLAWS")
        for s, sent in enumerate(self.flaws):
            print(f"{s+1}. {sent}")

        print("ITEMS")
        for i, item in enumerate(self.items):
            print(f"{i+1}. {item['name']}: {item['description']}")
            
    # Adding trait.
    def add_trait(self, trait):
        self.traits.append(trait)
    
    # Adding flaw.
    def add_flaw(self, flaw):
        self.flaws.append(flaw)
    
    # Adding item.
    def add_item(self, item_name, item_desc):
        self.items.append({'name': item_name, 'description': item_desc})

    # Removing trait.
    def remove_trait(self, idx):
        self.traits = self.traits[:idx] + self.traits[idx+1:]

    # Removing flaw.
    def remove_flaw(self, idx):
        self.flaws = self.flaws[:idx] + self.flaws[idx+1:]

    # Removing item.
    def remove_item(self, idx):
        self.items = self.items[:idx] + self.items[idx+1:]
