import openai
import torch


# The whole NPC class.
class NPC():
  def __init__(self, **init_params):
    # Model info.
    self.model_name = init_params['model_name']

    # Data processing info.,
    self.num_turns = init_params['num_turns']
    self.additional_contexts = init_params['additional_contexts']
    self.simple_concat = init_params['simple_concat']
    self.encoder_name = init_params['encoder_name']
    if not self.simple_concat:
      assert self.encoder_name is not None, "There should be the encoder name if it is a selective concatenation mode."

    # Actual game envrionment info.
    self.npc_name = init_params['npc_name']
    self.setting = init_params['setting']
    self.npc_persona = init_params['npc_persona']
    self.npc_goal = init_params['npc_goal']

    # History is stored as long as the instance is alive.
    self.history = []

    # HP for battle.
    self.npc_hp = init_params['npc_hp']

  # Refreshing history. (Just in case.)
  def refresh_history(self):
    self.history = []

  # Updating history after generation.
  def update_history(self, query, response, player_name, tokenizer=None, encoder=None):
    if self.simple_concat:
      self.history.append((player_name, query))
      self.history.append((self.npc_name, response))
    else:
      token_ids = torch.LongTensor(tokenizer(query)['input_ids']).unsqueeze(0).to(encoder.device)
      query_embs = torch.mean(encoder(token_ids).last_hidden_state, dim=1).detach()  # (1, d_h)
      self.history.append((player_name, query, query_embs[0]))

      token_ids = torch.LongTensor(tokenizer(response)['input_ids']).unsqueeze(0).to(encoder.device)
      response_embs = torch.mean(encoder(token_ids).last_hidden_state, dim=1).detach()  # (1, d_h)
      self.history.append((self.npc_name, response, response_embs[0]))

  # Default completion method.
  def generate_response(self, messages, **decoding_params):
    completion = openai.ChatCompletion.create(
      model=self.model_name,
      messages=messages,
      temperature=decoding_params['temp'] if 'temp' in decoding_params else 0.7,
      max_tokens=decoding_params['max_tokens'] if 'max_tokens' in decoding_params else 128,
      top_p=decoding_params['top_p'] if 'top_p' in decoding_params else 0.8,
      frequency_penalty=decoding_params['frequency_penalty'] if 'frequency_penalty' in decoding_params else 2,
      presence_penalty=decoding_params['presence_penalty'] if 'presence_penalty' in decoding_params else 0,
    )

    return completion['choices'][0]['message']['content']

  # Inference function for simple concatenation.
  # history: a list of tuple (character, utterance).
  def infer_simple_concat(self, query, player_name, player_persona, player_goal, **decoding_params):
    prompt = self.make_prompt(player_name, player_persona, player_goal, **decoding_params)

    new_hist = (player_name, query)
    updated_history = self.history + [new_hist]
    if self.num_turns == 'all':
      start = 0
    else:
      start = max(0, len(self.history)-self.num_turns)

    messages = [{'role': 'system', 'content': prompt}]
    history_messages = [{'role': 'user', 'content': hist[1]} if hist[0] == player_name else {'role': 'assistant', 'content': hist[1]} for hist in updated_history[start:]]
    messages += history_messages

    response = self.generate_response(messages, **decoding_params)
    return response

  # Inference function for selective concatenation.
  # history: a list of tuple (character, utterance, embedding).
  def infer_selective_concat(self, query, player_name, player_persona, player_goal, tokenizer, encoder, **decoding_params):
    prompt = self.make_prompt(player_name, player_persona, player_goal, **decoding_params)

    token_ids = torch.LongTensor(tokenizer(query)['input_ids']).unsqueeze(0).to(encoder.device)
    query_embs = torch.mean(encoder(token_ids).last_hidden_state, dim=1).detach()  # (1, d_h)
    new_history = (player_name, query, query_embs[0])

    if self.num_turns == 'all' or len(self.history) <= self.num_turns-1:
      updated_history = self.history + [new_history]
    else:
      cos = nn.CosineSimilarity(dim=1, eps=1e-6)
      cands = [hist[2] for hist in self.history]
      cands_embs = torch.stack(cands, dim=0)  # (B, d_h)
      sims = cos(query_embs.repeat(cands_embs.shape[0], 1), cands_embs)  # (B)
      selected_idxs = sorted(torch.topk(sims, self.num_turns-1).indices.tolist())  # (T-1)
      
      updated_history = [self.history[idx] for idx in selected_idxs] + [new_history]

    messages = [{'role': 'system', 'content': prompt}]
    history_messages = [{'role': 'user', 'content': hist[1]} if hist[0] == player_name else {'role': 'assistant', 'content': hist[1]} for hist in updated_history]
    messages += history_messages

    response = self.generate_response(messages, **decoding_params)
    return response

  # Making a system prompt to be fed into the ChatGPT.
  def make_prompt(self, player_name, player_persona, player_goal, **decoding_params):
    prompt = "You are an NPC for a fantasy text adventure game. " + \
      f"Your name is '{self.npc_name}' and the user's is '{player_name}'. " + \
      f"Generate a response in 1-2 sentences for the given dialogue history and additional information.{decoding_params['part_sep']}"
    if 'setting' in self.additional_contexts:
      setting_prompt = f"Setting: {' '.join(self.setting)}{decoding_params['part_sep']}"
      prompt += setting_prompt
    if 'persona' in self.additional_contexts:
      persona_prompt = f"Persona: {self.npc_name} - {' '.join(self.npc_persona)} {player_name} - {' '.join(player_persona)}{decoding_params['part_sep']}"
      prompt += persona_prompt
    if 'goal' in self.additional_contexts:
      goal_prompt = f"Goal: {self.npc_name} - {self.npc_goal} {player_name} - {player_goal}{decoding_params['part_sep']}"
      prompt += goal_prompt
    
    return prompt[:-1]