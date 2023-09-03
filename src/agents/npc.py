from kani import Kani
from kani.models import ChatRole, ChatMessage, FunctionCall
from kani.exceptions import FunctionCallException, NoSuchFunction, WrappedCallException
from torch import nn
from typing import AsyncIterable, Union
from src.models.embedding import EmbeddingModel
from src.models.kani_models import generate_engine

import torch
import logging

log = logging.getLogger('kani')

# The whole NPC class.
class NPC(Kani):
    def __init__(self, name: str, hp: int, num_turns: Union[str, int], concat_type:str, sent_emb_size: int=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Properties which are not part of kani system.
        self.name = name
        self.hp = hp
        self.num_turns = num_turns
        self.concat_type = concat_type
        if self.num_turns is None:
            self.num_turns = 'all'
        if self.concat_type is None:
            self.concat_type = 'simple'

        # Validating values.
        assert self.hp > 0, "The HP of the NPC should be larger than 0."
        assert self.num_turns == 'all' or (isinstance(self.num_turns, int) and self.num_turns > 0), "The number of turns should be 'all' or a positive integer."
        assert self.concat_type in ['simple', 'retrieval', 'summarization'], "The concatenation type should be 'simple', 'retrieval' or 'summarization'."
        
        if self.concat_type == 'retrieval':
            assert sent_emb_size is not None, "The size of hidden state vector for sentence embedding should be provided if the concatenation type is retrieval."
            self.sent_embs = torch.empty(0, sent_emb_size)
        elif self.concat_type == 'summarization':
            # This is an additional engine for summarization.
            # TODO: Supporting another engines & models.
            self.summ_engine = generate_engine(engine_name='openai', model_index="gpt-3.5-turbo")

    # Generating the chat prompt based on the concatenation type.
    async def get_prompt(self):
        if self.concat_type == 'simple':
            return self.get_simple_prompt(self.chat_history)
        elif self.concat_type == 'retrieval':
            return self.get_retrieval_prompt()
        elif self.concat_type == 'summarization':
            return await self.get_summarization_prompt()

    # Making a prompt by the simple concatenation rule.
    def get_simple_prompt(self, history: list[ChatMessage]):
        num_tokens_left = self.max_context_size - self.always_len
        if self.num_turns == 'all':
            num_turns = len(self.chat_history)
        else:
            num_turns = self.num_turns

        messages = []
        for message in reversed(history[-num_turns:]):
            num_tokens = self.message_token_len(message)
            num_tokens_left -= num_tokens
            if num_tokens_left > 0:
                messages.insert(0, message)
            else:
                break
        
        return self.always_included_messages + messages

    # Making a prompt by the retrieval concatenation rule.
    def get_retrieval_prompt(self):
        if self.num_turns == 'all' or self.num_turns >= len(self.chat_history):
            return self.get_simple_prompt(self.chat_history)

        query_emb = self.sent_embs[-1]  # (d_h)
        cands_embs = self.sent_embs[:-1]  # (N-1, d_h)
        
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        sims = cos(query_emb.unsqueeze(0).repeat(cands_embs.shape[0], 1), cands_embs)  # (N-1)
        selected_idxs = sorted(torch.topk(sims, self.num_turns-1).indices.tolist())  # (T-1)
        history = [self.chat_history[idx] for idx in selected_idxs] + [self.chat_history[-1]]
        return self.get_simple_prompt(history)

    # Making a prompt by the summarization rule.
    async def get_summarization_prompt(self):
        messages = self.get_simple_prompt(self.chat_history)
        messages = messages[len(self.always_included_messages):-1]  # Excluding always_included_messages for summarization.
        system_prompt = "You are an assistant to summarize given dialogue history within 10 sentences. " + \
            "Note that this is a conversation between the player and the NPC in a fantasy text role-playing game."
        summ_kani = Kani(self.summ_engine, chat_history=messages, system_prompt=system_prompt)
        summ = await summ_kani.chat_round_str("Please summarize the conversation so far.")
        return self.always_included_messages + [ChatMessage.assistant(summ, name="Summarizer"), self.chat_history[-1]]

    # Updating history before/after generation.
    def update_history(self, message: ChatMessage, embedding_model: EmbeddingModel=None):
        if self.concat_type == 'retrieval':
            assert embedding_model is not None, "The embedding model has not been provided for calculating the sentence embedding."
            new_emb = embedding_model.get_sentence_embedding(message.content)  # (d_h)
            self.sent_embs = torch.cat((self.sent_embs, new_emb), 0)  # (N, d_h)
        else:
            self.chat_history.append(message)

    # Overriding full_round for updating the chat history according to the concatenation policy.
    async def full_round(self, query: str, player_name: str, embedding_model: EmbeddingModel=None, **kwargs) -> AsyncIterable[ChatMessage]:
        retry = 0
        is_model_turn = True
        async with self.lock:
            self.update_history(ChatMessage.user(query, name=player_name), embedding_model=embedding_model)

            while is_model_turn:
                # do the model prediction
                completion = await self.get_model_completion(**kwargs)
                message = completion.message
                self.update_history(message, embedding_model=embedding_model)
                yield message

                # if function call, do it and attempt retry if it's wrong
                if not message.function_call:
                    return

                try:
                    is_model_turn = await self.do_function_call(message.function_call, embedding_model=embedding_model)
                except FunctionCallException as e:
                    should_retry = await self.handle_function_call_exception(message.function_call, e, retry)
                    # retry if we have retry attempts left
                    retry += 1
                    if not should_retry:
                        # disable function calling on the next go
                        kwargs = {**kwargs, "include_functions": False}
                    continue
                else:
                    retry = 0

    # Overriding do_function_call for updating the chat history according to the concatenation policy.
    async def do_function_call(self, call: FunctionCall, embedding_model: EmbeddingModel=None) -> bool:
        log.debug(f"Model requested call to {call.name} with data: {call.arguments!r}")
        # get func
        f = self.functions.get(call.name)
        if not f:
            raise NoSuchFunction(call.name)
        # call it
        try:
            result = await f(**call.kwargs)
            result_str = str(result)
            log.debug(f"{f.name} responded with data: {result_str!r}")
        except Exception as e:
            raise WrappedCallException(f.auto_retry, e) from e
        msg = ChatMessage.function(f.name, result_str)
        # if we are auto truncating, check and see if we need to
        if f.auto_truncate is not None:
            message_len = self.message_token_len(msg)
            if message_len > f.auto_truncate:
                log.warning(
                    f"The content returned by {f.name} is too long ({message_len} > {f.auto_truncate} tokens), auto"
                    " truncating..."
                )
                msg = self._auto_truncate_message(msg, max_len=f.auto_truncate)
                log.debug(f"Auto truncate returned {self.message_token_len(msg)} tokens.")
        # save the result to the chat history
        self.update_history(msg, embedding_model=embedding_model)
        # yield whose turn it is
        return f.after == ChatRole.ASSISTANT
