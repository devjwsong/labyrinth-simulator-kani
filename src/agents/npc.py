from models.chatgpt import ChatGPTModel
from torch import nn

import torch

# Selecting the generation model.
def set_model(model_name):
    if model_name == "ChatGPT":
        model = ChatGPTModel()

    return model

# The whole NPC class.
class NPC():
    def __init__(self, **init_params):
        # Basic specs.
        self.model = set_model(init_params['model_name'])
        self.hp = init_params['hp']

        # Additional context.
        self.name = init_params['name']
        self.setting = init_params['setting']
        self.persona = init_params['persona']
        self.goal = init_params['goal']

        # Data processing info.
        self.num_turns = init_params['num_turns']
        self.simple_concat = init_params['simple_concat']

        # History is stored as long as the instance is alive.
        self.history = []

    # Refreshing history. (Just in case.)
    def refresh_history(self):
        self.history = []

    # Updating history after generation.
    def update_history(self, query, response, player_name, embedding_model):
        if self.simple_concat:
            self.history.append((player_name, query, torch.Tensor(0.0)))
            self.history.append((self.name, response, torch.Tensor(0.0)))
        else:
            query_emb = embedding_model.get_sentence_embedding(query)  # (d_h)
            self.history.append((player_name, query, query_emb))

            response_emb = embedding_model.get_sentence_embedding(response)  # (d_h)
            self.history.append((self.npc_name, response, response_emb))

    # Inference function for simple concatenation.
    def infer_simple_concat(self, query, player_name, player_persona, player_goal, **decoding_params):
        prompt = self.make_prompt(player_name, player_persona, player_goal, **decoding_params)

        new_hist = (player_name, query, torch.Tensor(0.0))
        updated_history = self.history + [new_hist]
        if self.num_turns == 'all':
            start = 0
        else:
            start = max(0, len(self.history)-self.num_turns)

        messages = [{'role': 'system', 'content': prompt}]
        history_messages = [{'role': 'user', 'content': hist[1]} if hist[0] == player_name else {'role': 'assistant', 'content': hist[1]} for hist in updated_history[start:]]
        messages += history_messages

        response = self.model.generate_response(messages, **decoding_params)
        return response

    # Inference function for selective concatenation.
    def infer_selective_concat(self, query, player_name, player_persona, player_goal, embedding_model, **decoding_params):
        prompt = self.make_prompt(player_name, player_persona, player_goal, **decoding_params)

        query_emb = embedding_model.get_sentence_embedding(query)  # (d_h)
        new_history = (player_name, query, query_emb)

        if self.num_turns == 'all' or len(self.history) <= self.num_turns-1:
            updated_history = self.history + [new_history]
        else:
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            cands = [hist[2] for hist in self.history]
            cands_embs = torch.stack(cands, dim=0)  # (B, d_h)
            sims = cos(query_emb.unsqueeze(0).repeat(cands_embs.shape[0], 1), cands_embs)  # (B)
            selected_idxs = sorted(torch.topk(sims, self.num_turns-1).indices.tolist())  # (T-1)
            
            updated_history = [self.history[idx] for idx in selected_idxs] + [new_history]

        messages = [{'role': 'system', 'content': prompt}]
        history_messages = [{'role': 'user', 'content': hist[1]} if hist[0] == player_name else {'role': 'assistant', 'content': hist[1]} for hist in updated_history]
        messages += history_messages

        response = self.model.generate_response(messages, **decoding_params)
        return response

    # Making a system prompt to be fed into the generation model.
    def make_prompt(self, player_name, player_persona, player_goal, **decoding_params):
        prompt = "You are an NPC for a fantasy text adventure game. " + \
            f"Your name is '{self.name}' and the user's is '{player_name}'. " + \
            f"Generate a response in 1-2 sentences for the given dialogue history and additional information.{decoding_params['part_sep']}"

        setting_prompt = f"Setting: {' '.join(self.setting)}{decoding_params['part_sep']}"
        prompt += setting_prompt

        persona_prompt = f"Persona: {self.name} - {' '.join(self.persona)} {player_name} - {' '.join(player_persona)}{decoding_params['part_sep']}"
        prompt += persona_prompt

        goal_prompt = f"Goal: {self.name} - {self.goal} {player_name} - {player_goal}{decoding_params['part_sep']}"
        prompt += goal_prompt
        
        return prompt[:-1]
