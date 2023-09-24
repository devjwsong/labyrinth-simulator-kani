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
        print(f"<Character Sheet: {self.name}>")
        print(f"Name: {self.name}")
        print(f"Kin: {self.kin}")
        
        print("Persona")
        for s, sent in enumerate(self.persona):
            print(f"{s+1}. {sent}")

        print(f"Goal: {self.goal}")

        print("Traits")
        for s, sent in enumerate(self.traits):
            print(f"{s+1}. {sent}")

        print("Flaws")
        for s, sent in enumerate(self.flaws):
            print(f"{s+1}. {sent}")

        print("Items")
        for i, item in enumerate(self.items):
            print(f"{i+1}. {item['name']}: {item['description']}")

    # Confirming the modification.
    def confirm(self, desc):
        while True:
            res = input(f"{desc} ('y' or 'n')")
            if res == 'y':
                return True
            elif res == 'n':
                return False

            print("Your answer should be either 'y' or 'n'.")
            
    # Adding trait.
    def add_trait(self, trait):
        confirmed = self.confirm(desc=f"Are you sure to add this trait: {trait}?")
        if confirmed:
            self.traits.append(trait)
    
    # Adding flaw.
    def add_flaw(self, flaw):
        confirmed = self.confirm(desc=f"Are you sure to add this flaw: {flaw}?")
        if confirmed:
            self.flaws.append(flaw)
    
    # Adding item.
    def add_item(self, item_name, item_desc):
        confirmed = self.confirm(desc=f"Are you sure to add this item: ({item_name}: {item_desc})?")
        if confirmed:
            self.items.append({'name': item_name, 'description': item_desc})

    # Removing trait.
    def remove_trait(self, idx):
        confirmed = self.confirm(desc=f"Are you sure to remove this trait: {self.traits[idx]}?")
        if confirmed:
            self.traits = self.traits[:idx] + self.traits[idx+1:]

    # Removing flaw.
    def remove_flaw(self, idx):
        confirmed = self.confirm(desc=f"Are you sure to remove this flaw: {self.flaws[idx]}?")
        if confirmed:
            self.flaws = self.flaws[:idx] + self.flaws[idx+1:]

    # Removing item.
    def remove_item(self, idx):
        confirmed = self.confirm(desc=f"Are you sure to remove this item: ({self.items[idx]['name']}, {self.items[idx]['description']})?")
        if confirmed:
            self.items = self.items[:idx] + self.items[idx+1:]
