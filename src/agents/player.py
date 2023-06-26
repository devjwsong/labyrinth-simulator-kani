# The player class.
class Player():
  def __init__(self, **init_params):
    # Basic specs.
    self.hp = init_params['hp']
    self.actions = init_params['actions']
    self.items = init_params['items']
    
    # Additional context.
    self.name = init_params['name']
    self.setting = init_params['setting']
    self.persona = init_params['persona']
    self.goal = init_params['goal']

  # Adding a query.
  def make_query(self):
    query = input(f"{self.name}: ")
    return query

  # Listing the available actions.
  def list_actions(self):
    print("<Available actions and damages>")
    for idx, pair in self.actions.items():
      print(f"{idx}: {pair[0]} - {pair[1]}.")

  # Listing the available items.
  def list_items(self):
    print("<Available items with name, type and description>")
    for idx, triple in self.items.items():
      print(f"{idx}: {triple[0]}")
      print(f"type: {triple[1]}")
      print(f"desc: {triple[2]}")
