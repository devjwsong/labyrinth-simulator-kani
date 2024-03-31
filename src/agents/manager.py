from kani import Kani, ai_function, AIParam
from kani.models import ChatMessage, ChatRole, FunctionCall, ToolCall
from kani.exceptions import FunctionCallException, MessageTooLong, NoSuchFunction, WrappedCallException
from kani.internal import FunctionCallResult, ExceptionHandleResult
from kani.utils.message_formatters import assistant_message_contents
from kani.engines.base import BaseCompletion
from agents.player import Player, PlayerKani
from constants import (
    SEP,
    RULE_SUMMARY,
    STATE_UPDATE_PROMPT,
    VALIDATE_SUCCESS_PROMPT, 
    VALIDATE_FAILURE_PROMPT,
    DIFFICULTY_PROMPT,
    CREATE_NPC_PROMPT,
    OBTAINABLE_CHECK_PROMPT,
    TABLE_PROCESSING_PROMPT,
    EXPENDABLE_CHECK_PROMPT, 
    SUMMARIZE_PROMPT
)
from utils import (
    print_system_log, 
    remove_punctuation, 
    select_options, 
    select_random_options, 
    clean_history,
    convert_into_dict,
    convert_into_natural, 
    convert_into_number, 
    convert_into_class_idx
)
from typing import AsyncIterable, Annotated, Tuple, Callable
from argparse import Namespace
from copy import deepcopy
from itertools import chain
from sentence_transformers import SentenceTransformer, util

import json
import logging
import random
import numpy as np
import torch
import asyncio

log = logging.getLogger("kani")
message_log = logging.getLogger("kani.messages")


# The whole game manager class.
class GameManager(Kani):
    def __init__(self, scene: dict, main_args: Namespace, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Attributes which should be initialized before the game.
        self.chapter = scene['chapter']
        self.scene = scene['scene']
        self.scene_summary = scene['scene_summary']
        self.npcs = scene['npcs']
        self.success_condition = scene['success_condition']
        self.failure_condition = scene['failure_condition']
        self.game_flow = scene['game_flow']
        self.environment = scene['environment']
        self.random_tables = scene['random_tables']
        self.consequences = scene['consequences']

        # Additional arguments for prompt design policy.
        self.concat_policy = main_args.concat_policy
        self.max_num_msgs = main_args.max_num_msgs
        self.summarization = True if main_args.summarization else False
        self.summ_period = main_args.summ_period
        self.clear_raw_logs = True if main_args.clear_raw_logs else False

        # Additional attributes for enabling the prompt policies.
        self.encoder = None
        if main_args.concat_policy == 'retrieval' or main_args.rule_injection == 'retrieval':
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            self.encoder = SentenceTransformer('all-mpnet-base-v2').to(device)
        self.sent_embs = np.empty((0, self.encoder.get_sentence_embedding_dimension())) if self.concat_policy == 'retrieval' else None
        self.current_queries = []
        self.raw_history = []
        self.start_idx = 0
        self.turn_count = 0
        self.retrieved_messages = None
        self.retrieved_rules = None

        # Additional attributes for game play.
        self.players = []
        self.name_to_idx = {}
        self.is_action_scene = False
        self.gameplay_logs = []

        # Pre-buidling the rule prompt or embeddings.
        self.game_rules = []
        self.rule_embs = None
        if main_args.rule_injection == 'retrieval':
            self.game_rules = list(chain.from_iterable(RULE_SUMMARY))
            self.rule_embs = self.encoder.encode(self.game_rules).astype('float64')

            assert self.rule_embs.shape[0] == len(self.game_rules), "The number of rule embeddings should be identical to the length of rule list."

    # Setting the attributes in the scene.
    def set_scene(self, obj):
        self.chapter = obj['chapter']
        self.scene = obj['scene']
        self.scene_summary = obj['scene_summary']
        self.npcs = obj['npcs']
        self.success_condition = obj['success_condition']
        self.failure_condition = obj['failure_condition']
        self.game_flow = obj['game_flow']
        self.environment = obj['environment']
        self.random_tables = obj['random_tables']
        self.consequences = obj['consequences']

    # Setting the attributes in a player.
    def set_player(self, player: Player, obj: dict):
        player.name = obj['name']
        player.kin = obj['kin']
        player.persona = obj['persona']
        player.goal = obj['goal']
        player.traits = obj['traits']
        player.flaws = obj['flaws']
        player.inventory = obj['inventory']
        player.additional_notes = obj['additional_notes']

    # Getter for NPC in a natural format.
    def get_npc(self, info):
        return f"Kin: {info['kin']} {SEP} Persona: {', '.join(info['persona'])} {SEP} Goal: {info['goal']} {SEP} Trait: {info['trait']} {SEP} Flaw: {info['flaw']}"

    # Getter for NPCs in a (numbered) list format.
    def get_npcs(self, with_number=False):
        res = []
        for n, (name, info) in enumerate(self.npcs.items()):
            if with_number:
                res.append(f"({n+1}) Name: {name} {SEP} {self.get_npc(info)}")
            else:
                res.append(f"Name: {name} {SEP} {self.get_npc(info)}")
        return res

    # Getter for game flow in a (numbered) list format.
    def get_game_flow(self, with_number=False):
        if with_number:
            return [f"({f+1}) {flow}" for f, flow in enumerate(self.game_flow)]
        return self.game_flow

    # Getter for environment in a (numbered) list format.
    def get_environment(self, with_number=False):
        if with_number:
            return [f"({o+1}) {object}: {desc}" for o, (object, desc) in enumerate(self.environment.items())]
        return [f"{object}: {desc}" for object, desc in self.environment.items()]

    # Getter for random tables in a (numbered) list format.
    def get_random_tables(self, with_number=False):
        res = []
        for t, (table, entries) in enumerate(self.random_tables.items()):
            if with_number:
                res.append(f"({t+1}) {table}: {(' ' + SEP + ' ').join(entries)}")
            else:
                res.append(f"{table}: {(' ' + SEP + ' ').join(entries)}")
        return res

    # Showing the scene information which the manager has initialized.
    def show_scene(self):
        print("<CHAPTER>")
        print(self.chapter)

        print("<SCENE>")
        print(self.scene)

        print("<SCENE SUMMARY>")
        print('\n'.join(self.scene_summary))

        print("<NPCS>")
        print('\n'.join(self.get_npcs(with_number=True)))

        print("<SUCCESS CONDITION>")
        print(self.success_condition)

        print("<FAILURE CONDITION>")
        print(self.failure_condition)

        print("<GAME FLOW>")
        print('\n'.join(self.get_game_flow(with_number=True)))

        print("<ENVIRONMENT>")
        print('\n'.join(self.get_environment(with_number=True)))

        print("<RANDOM TABLES>")
        print('\n'.join(self.get_random_tables(with_number=True)))

        print("<CONSEQUENCES>")
        print(self.consequences)

    # Encoding the chat messages into the sentence embedding vectors.
    def encode_messages(self, messages: list[ChatMessage]):
        contents = [convert_into_natural(message) for message in messages]
        return self.encoder.encode(contents).astype('float64')  # (N, d)

    # Overriding add_to_history.
    async def add_to_history(self, messages: list[ChatMessage], store_in_raw: bool=True):
        self.chat_history += messages
        if store_in_raw:
            self.raw_history += messages

        # Sentence embedding for the retrieval.
        if self.sent_embs is not None:
            embs = self.encode_messages(messages)  # (N, d)
            self.sent_embs = np.concatenate((self.sent_embs, embs))

            # The number of sentence embeddings and chat logs should always be identical.
            assert len(self.chat_history) == self.sent_embs.shape[0], "The sentence embeddings and chat histories are not synced."

    # Making a prompt using the simple concatenation.
    def get_simple_history(self) -> list[ChatMessage]:
        valid_chat_history = self.chat_history + self.current_queries
        if self.max_num_msgs is None:
            return valid_chat_history
        else:
            if len(self.current_queries) > self.max_num_msgs:
                return deepcopy(self.current_queries)

            valid_chat_history = valid_chat_history[max(len(valid_chat_history)-self.max_num_msgs, 0):]

        return valid_chat_history

    # Making a prompt using the retrieval concatenation.
    def get_retrieval_history(self) -> list[ChatMessage]:
        # If this is the case, retrieval has no meaning.
        if len(self.chat_history) + len(self.current_queries) <= self.max_num_msgs or self.max_num_msgs <= len(self.current_queries):
            return self.get_simple_history()
        
        # Calculating the max-pooled cosine similarities.
        top_n = self.max_num_msgs - len(self.current_queries)
        query_embs, cand_embs = self.encode_messages(self.current_queries), self.sent_embs  # (Q, d), (C, d)
        cos_sims = util.cos_sim(query_embs, cand_embs)  # (Q, C)
        scores = torch.max(cos_sims, dim=0).values  # (C)

        # Sorting the candidate logs by the similarities.
        idxs = torch.sort(scores, descending=True).indices[:top_n]
        idxs = torch.sort(idxs).values
        retrieved, scores = [self.chat_history[idx] for idx in idxs], [scores[idx].item() for idx in idxs]
        valid_chat_history = retrieved + self.current_queries

        # Checking the length of the valid chat logs.
        assert len(valid_chat_history) == self.max_num_msgs, "The numbers of sampled chat logs and the maximum number of turns are different."
        assert len(retrieved) == top_n, "The number of retrieved messages is not same as the pre-defined top n."
        assert len(retrieved) == len(scores), "The retrieved messages are not matched with the calculated scores."

        # Since this function only works when concat_policy=retrieval, the retrieved messages are also exported.
        self.retrieved_messages = list(zip(retrieved, scores))

        return valid_chat_history

    # Summarizing the given dialogue history.
    async def summarize_history(self, input_history: list[ChatMessage]) -> ChatMessage:
        # The default system prompt for the instruction.
        system_prompt = ' '.join(SUMMARIZE_PROMPT)
        
        kani = Kani(self.engine, chat_history=input_history, system_prompt=system_prompt)
        generation_params = {
            'temperature': 0.5,
            'top_p': 1,
            'presence_penalty': 0,
            'frequency_penalty': 0,
        }
        res = await kani.chat_round_str("Give me the summarization of the chat history so far.", **generation_params)

        return ChatMessage.system(content=res, name="Summary")

    # Overriding get_prompt.
    async def get_prompt(self,
        include_rules: bool = True,
        include_scene_state: bool = True,
        include_player_states: bool = True,
    ) -> list[ChatMessage]:
        # First, setting the additional information.
        default_prompt = deepcopy(self.always_included_messages)

        rule_prompt_len = 0
        if include_rules:
            rule_prompt = self.make_rule_prompt()
            rule_prompt_len = self.message_token_len(rule_prompt)
            default_prompt.append(deepcopy(rule_prompt))

        scene_prompt_len = 0
        if include_scene_state:
            scene_prompt = self.make_scene_prompt()
            scene_prompt_len = self.message_token_len(scene_prompt)
            default_prompt.append(deepcopy(scene_prompt))
        
        player_prompts_len = 0
        if include_player_states:
            player_prompts = self.make_player_prompts()
            for message in player_prompts:
                player_prompts_len += self.message_token_len(message)
            default_prompt += player_prompts

        always_len = self.always_len + rule_prompt_len + scene_prompt_len + player_prompts_len  # Additional length for rule/scene information.

        # If summarization + no period, valid_chat_history is just one summary and the current query.
        if self.summarization and self.summ_period is None:
            summary = await self.summarize_history(self.chat_history)
            valid_chat_history = [summary] + self.current_queries
        else:
            if self.concat_policy == 'simple':
                valid_chat_history = self.get_simple_history()
            elif self.concat_policy == 'retrieval':
                valid_chat_history = self.get_retrieval_history()

        remaining = max_size = self.max_context_size - always_len
        total_tokens = 0
        to_keep = 0  # messages to keep from the end of chat history
        for message in reversed(valid_chat_history):
            # get and check the message's length
            message_len = self.message_token_len(message)
            if message_len > max_size:
                func_help = (
                    ""
                    if message.role != ChatRole.FUNCTION
                    else "You may set `auto_truncate` in the @ai_function to automatically truncate long responses.\n"
                )
                raise MessageTooLong(
                    "The chat message's size is longer than the allowed context window (after including system"
                    " messages, always included messages, and desired response tokens).\n"
                    f"{func_help}Content: {message.content[:100]}..."
                )
            # see if we can include it
            remaining -= message_len
            if remaining >= 0:
                total_tokens += message_len
                to_keep += 1
            else:
                break
        log.debug(
            f"get_prompt() returned {always_len + total_tokens} tokens ({always_len} always) in"
            f" {len(self.always_included_messages) + to_keep} messages"
            f" ({len(self.always_included_messages)} always)"
        )

        if not to_keep:
            return default_prompt
        prompt = default_prompt + valid_chat_history[-to_keep:]

        return prompt

    # Making the rule prompt.
    def make_rule_prompt(self, top_n: int=5):
        if self.rule_embs is not None:  # This means the manager using the retrieval-based rules.
            # Calculating the cosine similarities between the queries and rules.
            query_embs = self.encode_messages(self.current_queries)  # (Q, d)
            cos_sims = util.cos_sim(query_embs, self.rule_embs)  # (Q, C)
            scores = torch.max(cos_sims, dim=0).values  # (C)

            # Sorting the candidate logs by the similarities.
            idxs = torch.sort(scores, descending=True).indices[:top_n]
            idxs = torch.sort(idxs).values
            valid_rules, scores = [self.game_rules[idx] for idx in idxs], [scores[idx].item() for idx in idxs]

            assert len(valid_rules) == top_n, "The number of retrieved rules is not same as the pre-defined top_n."
            assert len(valid_rules) == len(scores), "The retrieved rules are not matched with the calculated scores."

            # Since this function only works when rule_injection=retrieval, the retrieved rules are also exported.
            self.retrieved_rules = list(zip(valid_rules, scores))

            rule_content = '\n'.join(valid_rules)

        else:
            rule_content = '\n'.join([' '.join(part) for part in RULE_SUMMARY])
        
        rule_prompt = ChatMessage.system(name="Game_Rules", content=rule_content)
        return rule_prompt

    # Making the scene prompt.
    def make_scene_prompt(self):
        content = f"chapter={self.chapter}, scene={self.scene}, scene_summary={self.scene_summary}, " + \
            f"npcs={self.npcs}, success_condition={self.success_condition}, failure_condition={self.failure_condition}, " + \
            f"game_flow={self.game_flow}, environment={self.environment}, random_tables={self.random_tables}, consequences={self.consequences}, " + \
            f"is_action_scene={self.is_action_scene}"
        
        scene_prompt = ChatMessage.system(name="Scene_State", content=content)
        return scene_prompt

    # Making one player prompt.
    def make_player_prompt(self, player: Player):
        content = f"name={player.name}, kin={player.kin}, persona={player.persona}, goal={player.goal}, " + \
            f"traits={player.traits}, flaws={player.flaws}, inventory={player.inventory}, additional_notes={player.additional_notes}"
        player_prompt = ChatMessage.system(name="Player_State", content=content)
        return player_prompt

    # Making the player prompts.
    def make_player_prompts(self):
        player_prompts = []
        for player in self.players:
            player_prompts.append(self.make_player_prompt(player))

        return player_prompts

    # Making the context for exporting data.
    def make_context(self):
        context = {
            "scene": {
                "chapter": self.chapter,
                "scene": self.scene,
                "scene_summary": deepcopy(self.scene_summary),
                "npcs": deepcopy(self.npcs),
                "success_condition": self.success_condition,
                "failure_condition": self.failure_condition,
                "game_flow": deepcopy(self.game_flow),
                "environment": deepcopy(self.environment),
                "random_tables": deepcopy(self.random_tables),
                "consequences": self.consequences,
                "is_action_scene": self.is_action_scene
            },
        }
        players = []
        for player in self.players:
            players.append({
                "name": player.name,
                "kin": player.kin,
                "persona": deepcopy(player.persona),
                "goal": player.goal,
                "traits": deepcopy(player.traits),
                "flaws": deepcopy(player.flaws),
                "inventory": deepcopy(player.inventory),
                "additional_notes": deepcopy(player.additional_notes)
            })
        context["players"] = players

        return context

    # Overriding get_model_completion.
    async def get_model_completion(self, 
        include_functions: bool = True, 
        include_rules: bool = True,
        include_scene_state: bool = True,
        include_player_states: bool = True,
        **kwargs
    ) -> Tuple[BaseCompletion, list[ChatMessage]]:
        """Get the model's completion with the current chat state.

        Compared to :meth:`chat_round` and :meth:`full_round`, this lower-level method does not save the model's reply
        to the chat history or mutate the chat state; it is intended to help with logging or to repeat a call multiple
        times.

        :param include_functions: Whether to pass this kani's function definitions to the engine.
        :param kwargs: Arguments to pass to the model engine.
        """
        # get the current chat state
        messages = await self.get_prompt(include_rules, include_scene_state, include_player_states)

        # log it (message_log includes the number of messages sent and the last message)
        n_messages = len(messages)
        if n_messages == 0:
            message_log.debug("[0]>>> [requested completion with no prompt]")
        else:
            message_log.debug(f"[{n_messages}]>>> {messages[-1]}")

        # get the model's completion at the given state
        if include_functions:
            completion = await self.engine.predict(messages=messages, functions=list(self.functions.values()), **kwargs)
        else:
            completion = await self.engine.predict(messages=messages, **kwargs)

        # cache its length (if the completion isn't saved to state, this weakrefs and gc's later)
        message = completion.message
        self._message_tokens[message] = completion.completion_tokens or self.message_token_len(message)
        # and log it too
        message_log.debug(f"<<< {message}")
        return completion, messages

    # Overriding full_round.
    async def full_round(self, queries: list[ChatMessage], **kwargs) -> AsyncIterable[ChatMessage]:
        """Perform a full chat round (user -> model [-> function -> model -> ...] -> user).

        Yields each of the model's ChatMessages. A ChatMessage must have at least one of (content, function_call).

        Use this in an async for loop, like so::

            async for msg in kani.full_round("How's the weather?"):
                print(msg.content)

        :param query: The content of the user's chat message.
        :param kwargs: Additional arguments to pass to the model engine (e.g. hyperparameters).
        """

        retry = 0
        is_model_turn = True
        async with self.lock:
            self.current_queries = deepcopy(queries)
            generate_states = kwargs['generate_states']
            kwargs.pop('generate_states')

            while is_model_turn:
                # do the model prediction
                completion, messages = await self.get_model_completion(**kwargs)

                # Recording the current state.
                context = self.make_context()

                context["past_history"] = []
                for msg in self.raw_history:
                    context["past_history"].append(convert_into_dict(msg))
                context["current_queries"] = []
                for msg in self.current_queries:
                    context["current_queries"].append(convert_into_dict(msg))
                context["actual_prompt"] = []
                for msg in messages:
                    context["actual_prompt"].append(convert_into_natural(msg))
                if self.retrieved_messages is not None:
                    context["retrieved_messages"] = []
                    for msg, score in self.retrieved_messages:
                        context["retrieved_messages"].append([convert_into_natural(msg), score])
                    self.retrieved_messages = None
                if self.retrieved_rules is not None:
                    context["retrieved_rules"] = []
                    for rule, score in self.retrieved_rules:
                        context["retrieved_rules"].append([rule, score])
                    self.retrieved_rules = None

                message = completion.message
                if not message.tool_calls:
                    message = ChatMessage.assistant(name="Goblin_King", content=message.content)
                yield message

                # In the current query, all types of messages are stored.
                self.current_queries.append(message)
                context["generated"] = convert_into_dict(message)

                if not message.tool_calls:  # If there is no function call, this is the end of a turn.
                    self.gameplay_logs.append(context)

                    # If specified, the model updates the game states on its own.
                    if generate_states:
                        await self.update_states(deepcopy(self.current_queries))

                    break

                # run each tool call in parallel
                async def _do_tool_call(tc: ToolCall):
                    try:
                        return await self.do_function_call(tc.function, tool_call_id=tc.id)
                    except FunctionCallException as e:
                        return await self.handle_function_call_exception(tc.function, e, retry, tool_call_id=tc.id)

                # If this is a tool call, run the functions and gather the results.
                context['function_results'] = []
                is_model_turn = False
                should_retry_call = False
                n_errs = 0
                results = await asyncio.gather(*(_do_tool_call(tc) for tc in message.tool_calls))
                for result in results:                    
                    if isinstance(result, ExceptionHandleResult):
                        is_model_turn = True
                        n_errs += 1
                        # retry if any function says so
                        should_retry_call = should_retry_call or result.should_retry
                    else:
                        result, arguments, intermediate_res = result

                        # save the result to the chat history as a system message.
                        self.current_queries.append(result.message)

                        # Recording the function execution specifications.
                        context['function_results'].append({
                            'result': convert_into_dict(result.message),
                            'arguments': deepcopy(arguments),
                            'intermediate_results': deepcopy(intermediate_res)
                        })

                        # allow model to generate response if any function says so
                        is_model_turn = is_model_turn or result.is_model_turn

                # if we encountered an error, increment the retry counter and allow the model to generate a response
                if n_errs:
                    retry += 1
                    if not should_retry_call:
                        # disable function calling on the next go
                        kwargs["include_functions"] = False
                else:
                    retry = 0

                self.gameplay_logs.append(context)

            # After finishing the turn, the current queries should be added to the chat history.
            await self.add_to_history(clean_history(self.current_queries))

            # Increasing the turn count. If the summarization period has been reached, adding the summary.
            self.turn_count += 1
            if self.summarization and self.summ_period is not None and self.turn_count == self.summ_period:
                input_history = self.chat_history[self.start_idx:]
                summary = await self.summarize_history(input_history)
                await self.add_to_history([summary], store_in_raw=False)

                if self.clear_raw_logs:
                    self.chat_history = self.chat_history[:self.start_idx] + self.chat_history[-1:]
                    
                    if self.sent_embs is not None:
                        self.sent_embs = np.concatenate((self.sent_embs[:self.start_idx], self.sent_embs[-1:]), axis=0)

                        assert len(self.chat_history) == self.sent_embs.shape[0], "The sentence embeddings and chat histories are not synced."
                
                self.start_idx = len(self.chat_history)
                self.turn_count = 0

    # Overriding do_function_call.
    async def do_function_call(self, call: FunctionCall, tool_call_id: str = None) -> FunctionCallResult:
        """Resolve a single function call.

        By default, any exception raised from this method will be an instance of a :class:`.FunctionCallException`.

        You may implement an override to add instrumentation around function calls (e.g. tracking success counts
        for varying prompts). See :ref:`do_function_call`.

        :returns: A :class:`.FunctionCallResult` including whose turn it is next and the message with the result of the
            function call.
        :raises NoSuchFunction: The requested function does not exist.
        :raises WrappedCallException: The function raised an exception.
        """
        log.debug(f"Model requested call to {call.name} with data: {call.arguments!r}")
        # get func
        f = self.functions.get(call.name)
        if not f:
            raise NoSuchFunction(call.name)
        # call it
        try:
            result, arguments, intermediate_res = await f(**call.kwargs)
            result_str = str(result)
            log.debug(f"{f.name} responded with data: {result_str!r}")
        except Exception as e:
            raise WrappedCallException(f.auto_retry, e) from e
        msg = ChatMessage.function(f.name, result_str, tool_call_id=tool_call_id)
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

        return FunctionCallResult(is_model_turn=f.after == ChatRole.ASSISTANT, message=msg), arguments, intermediate_res

    # Overriding full_round_str.
    async def full_round_str(
        self,
        queries: list[ChatMessage],
        message_formatter: Callable[[ChatMessage], str | None] = assistant_message_contents,
        **kwargs,
    ) -> AsyncIterable[tuple[str, ChatRole]]:
        """Like :meth:`full_round`, but each yielded element is a str rather than a ChatMessage.

        :param queries: The list of the user's chat messages.
        :param message_formatter: A function that returns a string to yield for each message. By default, `
            `full_round_str`` yields the content of each assistant message.
        :param kwargs: Additional arguments to pass to the model engine (e.g. hyperparameters).
        """
        async for message in self.full_round(queries, **kwargs):
            if text := message_formatter(message):
                yield text

    # Updating the game state after every generation.
    async def update_states(self, current_queries: list[ChatMessage]):
        system_prompt = ' '.join(STATE_UPDATE_PROMPT)
        rule_content = '\n'.join([' '.join(part) for part in RULE_SUMMARY])
        system_prompt = f"{system_prompt}\n\nGame Rules: {rule_content}"

        kani = Kani(self.engine, chat_history=current_queries, system_prompt=system_prompt)
        generation_params = {
            'temperature': 0,
            'top_p': 1,
            'presence_penalty': 0,
            'frequency_penalty': 0,
        }

        # Updating the scene.
        prev_scene = self.make_scene_prompt()
        scene_res = await kani.chat_round_str(
            f"Generate the updated scene state from the previous scene state considering the given interaction.\n\nPrevious Scene State: {prev_scene.content}",
            **generation_params
        )

        try:
            new_scene_state = json.loads(scene_res)
            self.npcs = new_scene_state['npcs']
            self.environment = new_scene_state['environment']
            self.random_tables = new_scene_state['random_tables']
            self.is_action_scene = new_scene_state['is_action_scene']

        except json.decoder.JSONDecodeError as e:
            log.debug(scene_res)
            log.error(f"{e}: The output format cannot be converted into dict.")
            raise Exception()

        # Updating the players.
        for player in self.players:
            prev_state = self.make_player_prompt(player)
            player_res = await kani.chat_round_str(
                f"Generate the updated player state from the previous player state considering the given interaction.\n\nPrevious Player State: {prev_state.content}",
                **generation_params
            )

            try:
                new_player_state = json.loads(player_res)
                player.traits = new_player_state['traits']
                player.flaws = new_player_state['flaws']
                player.inventory = new_player_state['inventory']
            except json.decoder.JSONDecodeError as e:
                log.debug(scene_res)
                log.error(f"{e}: The output format cannot be converted into dict.")
                raise Exception()

        await kani.engine.close()         

    # Kani's function call for a dice roll test.
    @ai_function
    async def activate_test(self, 
        player_name: Annotated[str, AIParam(desc="The name of the player charater who should do the test.")],
        initial_difficulty: Annotated[int, AIParam(desc="The initially set difficulty of the task in a range of 2 and 6.")],
        final_difficulty: Annotated[int, AIParam(desc="The final difficulty which has been reduced by the teamwork of the party and the minimum should still be 2.")]
    ):
        """
        Activate a test if a player is trying to do something challenging with a certain difficulty. 
        Determine the original difficulty of the task first and then set the final difficulty after reducing it depending on the teamwork from other players. 
        If the samples returned from `use_random_table` function are related to a dice roll testing, call this function after sampling.
        """
        arguments = {'player_name': player_name, 'initial_difficulty': initial_difficulty, 'final_difficulty': final_difficulty}

        # Wrong argument: player_name does not exist.
        if player_name not in self.name_to_idx:
            msg = f"THE PLAYER NAME {player_name} CANNOT BE FOUND."
            print_system_log(msg, after_break=True)
            return msg, arguments, None

        player = self.players[self.name_to_idx[player_name]]

        # The default system prompt consists of the instruction.
        system_prompt = ' '.join(DIFFICULTY_PROMPT)
        player_prompt = self.make_player_prompt(player)
        system_prompt = f"{system_prompt}\n\nPlayer State: {player_prompt.content}"
        
        options = ["The test becomes easier.", "The test becomes harder.", "There is no change."]
        options_str = '\n'.join([f"{o}: {option}" for o, option in enumerate(options)])
        kani = Kani(self.engine, chat_history=clean_history(self.current_queries), system_prompt=system_prompt)
        generation_params = {
            'temperature': 0.2,
            'top_p': 1,
            'presence_penalty': 0,
            'frequency_penalty': 0,
        }

        res = await kani.chat_round_str(f"Would the test become easier, harder, or none of them depending on the player trait, flaw or item?\n\n{options_str}", **generation_params)
        res = convert_into_class_idx(res, options)

        intermediate_res = {f"Improvement/Hinderance of the test due to the player traits/flaws": options[res]}

        if res == 2:  # The difficulty is not affected.
            if not isinstance(player, PlayerKani):
                _ = input(f"THE TEST DIFFICULTY: {final_difficulty}: PRESS ANY KEY TO ROLL A DICE.")
            dice_result = random.randint(1, 6)

        elif res == 0:  # The test is improved.
            print_system_log("A TRAIT OR AN ITEM IN THE PLAYER MAKES THE TEST EASIER. YOU ROLL TWO DICES AND TAKE THE LARGER ONE.")
            if not isinstance(player, PlayerKani):
                _ = input(f"THE TEST DIFFICULTY: {final_difficulty}: PRESS ANY KEY TO ROLL TWO DICES.")
            result1, result2 = random.randint(1, 6), random.randint(1, 6)
            dice_result = max(result1, result2)
            print_system_log(f"RESULT 1 ({result1}) vs RESULT 2 ({result2}) => THE PLAYER GOT {dice_result}.")

        elif res == 1:  # The test is hindered.
            print_system_log("A FLAW IN THE PLAYER MAKES THE TEST HARDER. YOU ROLL TWO DICES AND TAKE THE SMALLER ONE.")
            if not isinstance(player, PlayerKani):
                _ = input(f"THE TEST DIFFICULTY: {final_difficulty}: PRESS ANY KEY TO ROLL TWO DICES.")
            result1, result2 = random.randint(1, 6), random.randint(1, 6)
            dice_result = min(result1, result2)
            print_system_log(f"RESULT 1 ({result1}) vs RESULT 2 ({result2}) => THE PLAYER GOT {dice_result}.")

        if dice_result < final_difficulty:
            msg = f"{player_name} HAS FAILED THE TEST. THE DICE ROLL RESULT {dice_result} IS SMALLER THAN THE DIFFICULTY {final_difficulty}."
        else:
            msg = f"{player_name} HAS SUCCEEDED THE TEST. THE DICE ROLL RESULT {dice_result} IS LARGER THAN OR EQUAL TO THE DIFFICULTY {final_difficulty}."

        # Updating the new chat message.
        print_system_log(msg, after_break=True)
        return msg, arguments, intermediate_res
            

    # Kani's function call for starting an action scene.
    @ai_function
    def activate_action_scene(self):
        """
        Activate an action scene if this is a circumstance that players should take action within a tight time limit. 
        If the samples returned from `use_random_table` function are related to starting a new action scene, call this function after sampling. 
        Do not call this function if action_scene is set to True already.
        """
        arguments, intermediate_res = None, None

        self.is_action_scene = True
        msg = "ACTION SCENE ACTIVATED."
        print_system_log(msg, after_break=True)
        return msg, arguments, intermediate_res

    # Kani's function call for ending an action scene.
    @ai_function
    def terminate_action_scene(self):
        """
        Terminate the current ongoing action scene if the urgent circumstance has been finished. 
        If the samples returned from `use_random_table` function are related to ending an ongoing action scene, call this function after sampling. 
        Do not call this function if action_scene is set to False already.
        """
        arguments, intermediate_res = None, None

        self.is_action_scene = False
        msg = "ACTION SCENE TERMINATED."
        print_system_log(msg, after_break=True)
        return msg, arguments, intermediate_res

    # Kani's function call for creating an NPC immediately.
    @ai_function
    async def create_npc(self, 
        npc_name: Annotated[str, AIParam(desc="The name of the NPC which should be set into the scene.")],
        npc_desc: Annotated[str, AIParam(desc="The additional description of the NPC.")]
    ):
        """
        Create an NPC if the player party encounters or requests to interact with an NPC which has not been initialized in the scene yet. 
        If there is a description of the NPC which should be included, pass it as a function parameter too. 
        If the samples returned from `use_random_table` function are related to setting a new NPC into the scene, call this function after sampling. 
        Do not call this function if the NPC already exists in the scene.
        """ 
        arguments = {'npc_name': npc_name, 'npc_desc': npc_desc}

        # Wrong activation/argument: The NPC already exists.
        if npc_name in self.npcs:
            msg = f"NPC {npc_name} ALREADY EXISTS."
            print_system_log(msg, after_break=True)
            return msg, arguments, None

        # The default system prompt consists of the instruction and the requirement for an NPC.
        system_prompt = ' '.join(CREATE_NPC_PROMPT)
        scene_prompt = self.make_scene_prompt()
        system_prompt = f"{system_prompt}\n\nScene State: {scene_prompt.content}"
        
        kani = Kani(self.engine, system_prompt=system_prompt)
        generation_params = {
            'temperature': 0,
            'top_p': 1,
            'presence_penalty': 0,
            'frequency_penalty': 0,
        }

        res = await kani.chat_round_str(f"Generate the specifications of the requested NPC.\n\nNPC name: '{npc_name}'\nAdditional description: {npc_desc}", **generation_params)

        # Converting & Fetching information.
        try:
            res = json.loads(res)

            assert isinstance(res['kin'], str), "THE KIN OF AN NPC IS NOT THE STRING TYPE."
            assert isinstance(res['persona'], list), "THE PERSONA OF AN NPC IS NOT THE LIST TYPE."
            assert isinstance(res['goal'], str), "THE GOAL OF AN NPC IS NOT THE STRING TYPE."
            assert isinstance(res['trait'], str), "THE TRAITS OF AN NPC IS NOT THE STRING TYPE."
            assert isinstance(res['flaw'], str), "THE FLAWS OF AN NPC IS NOT THE STRING TYPE."

            self.npcs[npc_name] = res

            intermediate_res = {f"Generated information of the NPC '{npc_name}'": res}

        except json.decoder.JSONDecodeError as e:
            log.debug(res)
            log.error(f"{e}: The output format cannot be converted into dict.")
            raise Exception()
        except KeyError as e:
            log.debug(res)
            log.error(f"{e}: Missing key.")
            raise Exception()

        msg = f"THE NPC {npc_name} HAS BEEN SET INTO THE SCENE."
        print_system_log(msg, after_break=True)
        return msg, arguments, intermediate_res

    # Kani's function call for adding a trait to the player.
    @ai_function
    def add_trait(self,
        player_name: Annotated[str, AIParam(desc="The name of the player charater whose new trait should be added.")],
        trait_name: Annotated[str, AIParam(desc="The name of the new trait to be added in the player.")],
        trait_desc: Annotated[str, AIParam(desc="The description of the trait.")]
    ):
        """
        Add a new trait as a player property if any circumstance necessitates it. 
        Pass the description of the trait as a parameter too if it exists. 
        If it does not exist, generate a brief description in one or two sentences. 
        If the samples returned from `use_random_table` function are related to adding a new trait to a player, call this function after sampling.
        Do not call this function if the trait already exists in the player who triggered this function.
        """
        arguments = {'player_name': player_name, 'trait_name': trait_name, 'trait_desc': trait_desc}
        
        # Wrong argument: The player does not exist.
        if player_name not in self.name_to_idx:
            msg = f"THE PLAYER NAME {player_name} CANNOT BE FOUND."
            print_system_log(msg, after_break=True)
            return msg, arguments, None

        player = self.players[self.name_to_idx[player_name]]

        # Wrong activation/argument: The trait already exists.
        if trait_name in player.traits:
            msg = f"THE PLAYER {player_name} ALREADY HAS THE TRAIT {trait_name}."
            print_system_log(msg, after_break=True)
            return msg, arguments, None

        player.add_trait(trait_name, trait_desc)

        msg = f"THE TRAIT {trait_name} HAS BEEN ADDED TO THE PLAYER {player_name}."
        updated_res = '\n'.join(player.get_traits(with_number=True))
        print_system_log(f"PLAYER TRAITS UPDATED:\n{updated_res}", after_break=True)
        print_system_log(msg, after_break=True)
        return msg, arguments, None

    # Kani's function call for adding a flaw to the player.
    @ai_function
    def add_flaw(self,
        player_name: Annotated[str, AIParam(desc="The name of the player charater whose new flaw should be added.")],
        flaw_name: Annotated[str, AIParam(desc="The name of the new flaw to be added in the player.")],
        flaw_desc: Annotated[str, AIParam(desc="The description of the flaw.")]
    ):
        """
        Add a new flaw as a player property if any circumstance necessitates it. 
        Pass the description of the flaw as a parameter too if it exists. 
        If it does not exist, generate a brief description in one or two sentences. 
        If the samples returned from `use_random_table` function are related to adding a new flaw to a player, call this function after sampling.
        Do not call this function if the flaw already exists in the player who triggered this function.
        """
        arguments = {'player_name': player_name, 'flaw_name': flaw_name, 'flaw_desc': flaw_desc}

        # Wrong argument: The player does not exist.
        if player_name not in self.name_to_idx:
            msg = f"THE PLAYER NAME {player_name} CANNOT BE FOUND."
            print_system_log(msg, after_break=True)
            return msg, arguments, None

        player = self.players[self.name_to_idx[player_name]]

        # Wrong activation/argument: The flaw already exists.
        if flaw_name in player.flaws:
            msg = f"THE PLAYER {player_name} ALREADY HAS THE FLAW {flaw_name}."
            print_system_log(msg, after_break=True)
            return msg, arguments, None

        player.add_flaw(flaw_name, flaw_desc)

        msg = f"THE FLAW {flaw_name} HAS BEEN ADDED TO THE PLAYER {player_name}."
        updated_res = '\n'.join(player.get_flaws(with_number=True))
        print_system_log(f"PLAYER FLAWS UPDATED:\n{updated_res}", after_break=True)
        print_system_log(msg, after_break=True)
        return msg, arguments, None

    # Kani's function call for adding an item to the player inventory.
    @ai_function
    def add_item(self,
        player_name: Annotated[str, AIParam(desc="The name of the player charater whose new item should be added.")],
        item_name: Annotated[str, AIParam(desc="The name of the new item to be added in the player.")],
        item_desc: Annotated[str, AIParam(desc="The description of the item.")]
    ):
        """
        Add a new item to a player inventory if any circumstance necessitates it. 
        Pass the description of the item as a parameter too if it exists. 
        If it does not exist, generate a brief description in one or two sentences. 
        If the samples returned from `use_random_table` function are related to adding a new item to a player, call this function after sampling.
        Do not call this function if the item already exists in the inventory of the player who triggered this function.
        """
        arguments = {'player_name': player_name, 'item_name': item_name, 'item_desc': item_desc}

        # Wrong argument: The player does not exist.
        if player_name not in self.name_to_idx:
            msg = f"THE PLAYER NAME {player_name} CANNOT BE FOUND."
            print_system_log(msg, after_break=True)
            return msg, arguments, None

        player = self.players[self.name_to_idx[player_name]]

        # Wrong activation/argument: The flaw already exists.
        if item_name in player.inventory:
            msg = f"THE PLAYER {player_name} ALREADY HAS THE ITEM {item_name}."
            print_system_log(msg, after_break=True)
            return msg, arguments, None

        obtain_msg = self.put_item(player, item_name, item_desc)

        intermediate_res = {}

        if item_name in player.inventory:
            intermediate_res[f"The item '{item_name}' added"] = True
        else:
            intermediate_res[f"The item '{item_name}' added"] = False
        msg = f"THE PLAYER {player_name} FOUND THE ITEM {item_name}.\n{obtain_msg}"
        print_system_log(msg, after_break=True)

        return msg, arguments, intermediate_res

    # Logic for putting an item into the player's inventory. (Not an AI function!)
    def put_item(self, player: Player, item_name: str, item_desc: str):
        # A sub logic that adds the item.
        def sub_logic(player, item_name, item_desc):
            player.add_item(item_name, item_desc)
            msg = f"THE ITEM {item_name} HAS BEEN ADDED TO THE INVENTORY OF THE PLAYER {player.name}."
            updated_res = '\n'.join(player.get_inventory(with_number=True))
            print_system_log(f"PLAYER INVENTORY UPDATED:\n{updated_res}", after_break=True)
            return msg

        if len(player.inventory) >= 6:  # The player inventory is already full.
            print_system_log("YOUR INVENTORY IS ALREADY FULL. CHOOSE ONE ITEM TO DISCARD FROM THE INVENTORY OR DECIDE NOT TO TAKE THE CURRENT ITEM.")
            options = [
                "Discarding one item from the inventory.",
                "Not taking the found item."
            ]
            selected = select_random_options(options) if isinstance(player, PlayerKani) else select_options(options)
            if selected == 0:  # Discarding any item from the inventory.
                print_system_log("WHICH ITEM ARE YOU GOING TO DISCARD?")
                selected = select_random_options(player.get_inventory()) if isinstance(player, PlayerKani) else select_options(player.get_inventory())
                removal_target = list(player.inventory.keys())[selected]
                remove_msg = self.remove_item(player.name, removal_target)

                msg = sub_logic(player, item_name, item_desc)
                msg = f"{remove_msg}\n{msg}"

                return msg
            
            # Not taking the found item.
            msg = f"THE PLAYER {player.name} DECIDED NOT TO TAKE THE ITEM {item_name}."

            return msg

        # Taking the item since there is still a room in the inventory.
        msg = sub_logic(player, item_name, item_desc)

        return msg

    # Kani's function call for removing a trait from the player.
    @ai_function
    def remove_trait(self,
        player_name: Annotated[str, AIParam(desc="The name of the player charater whose trait should be removed.")],
        trait_name: Annotated[str, AIParam(desc="The name of the trait which should be removed from the player.")]
    ):
        """
        Remove a trait from a player if any circumstance necessitates it. 
        If the samples returned from `use_random_table` function are related to removing a trait from a player, call this function after sampling.
        Do not call this function if the trait does not exist in the player who triggered this function.
        """
        arguments = {'player_name': player_name, 'trait_name': trait_name}

        # Wrong argument: The player does not exist.
        if player_name not in self.name_to_idx:
            msg = f"THE PLAYER NAME {player_name} CANNOT BE FOUND."
            print_system_log(msg, after_break=True)
            return msg, arguments, None

        player = self.players[self.name_to_idx[player_name]]

        # Wrong activation/argument: The trait does not exist.
        if trait_name not in player.traits:
            msg = f"THE PLAYER {player_name} DOES NOT HAVE THE TRAIT {trait_name}."
            print_system_log(msg, after_break=True)
            return msg, arguments, None

        # Removing the trait from the player.
        player.remove_trait(trait_name)

        msg = f"THE TRAIT {trait_name} HAS BEEN REMOVED FROM THE PLAYER {player_name}."
        updated_res = '\n'.join(player.get_traits(with_number=True))
        print_system_log(f"PLAYER TRAITS UPDATED:\n{updated_res}", after_break=True)
        print_system_log(msg, after_break=True)

        return msg, arguments, None

    # Kani's function call for removing a flaw from the player.
    @ai_function
    def remove_flaw(self,
        player_name: Annotated[str, AIParam(desc="The name of the player charater whose flaw should be removed.")],
        flaw_name: Annotated[str, AIParam(desc="The name of the flaw which should be removed from the player.")]
    ):
        """
        Remove a flaw from a player if any circumstance necessitates it. 
        If the samples returned from `use_random_table` function are related to removing a flaw from a player, call this function after sampling.
        Do not call this function if the flaw does not exist in the player who triggered this function.
        """
        arguments = {'player_name': player_name, 'flaw_name': flaw_name}

        # Wrong argument: The player does not exist.
        if player_name not in self.name_to_idx:
            msg = f"THE PLAYER NAME {player_name} CANNOT BE FOUND."
            print_system_log(msg, after_break=True)
            return msg, arguments, None

        player = self.players[self.name_to_idx[player_name]]

        # Wrong activation/argument: The flaw does not exist.
        if flaw_name not in player.flaws:
            msg = f"THE PLAYER {player_name} DOES NOT HAVE THE FLAW {flaw_name}."
            print_system_log(msg, after_break=True)
            return msg, arguments, None

        # Removing the flaw from the player.
        player.remove_flaw(flaw_name)

        msg = f"THE FLAW {flaw_name} HAS BEEN REMOVED FROM THE PLAYER {player_name}."
        updated_res = '\n'.join(player.get_flaws(with_number=True))
        print_system_log(f"PLAYER FLAWS UPDATED:\n{updated_res}", after_break=True)
        print_system_log(msg, after_break=True)

        return msg, arguments, None

    # Kani's function call for removing an item in the player's inventory.
    @ai_function
    def remove_item(self,
        player_name: Annotated[str, AIParam(desc="The name of the player charater who wants to remove the item from the inventory.")],
        item_name: Annotated[str, AIParam(desc="The name of the item which the player wants to discard.")]
    ):
        """
        Remove an item from a player's inventory if any circumstance necessitates it. 
        If the samples returned from `use_random_table` function are related to removing an item from a player, call this function after sampling. 
        Do not call this function if the item does not exist in the inventory of the player who triggered this function.
        """
        arguments = {'player_name': player_name, 'item_name': item_name}

        # Wrong argument: The player does not exist.
        if player_name not in self.name_to_idx:
            msg = f"THE PLAYER NAME {player_name} CANNOT BE FOUND."
            print_system_log(msg, after_break=True)
            return msg, arguments, None

        player = self.players[self.name_to_idx[player_name]]

        # Wrong activation/argument: The item does not exist.
        if item_name not in player.inventory:
            msg = f"THE PLAYER {player_name} DOES NOT HAVE THE ITEM {item_name}."
            print_system_log(msg, after_break=True)
            return msg, arguments, None

        # Removing the item from the inventory.
        desc = player.inventory[item_name]
        player.remove_item(item_name)

        # Discarded item is placed in the environment.
        self.environment[item_name] = desc

        msg = f"THE ITEM {item_name} HAS BEEN REMOVED FROM THE INVENTORY OF THE PLAYER {player_name}."
        updated_res = '\n'.join(player.get_inventory(with_number=True))
        print_system_log(f"PLAYER INVENTORY UPDATED:\n{updated_res}", after_break=True)
        print_system_log(msg, after_break=True)

        return msg, arguments, None

    # Kani's function call for using an item.
    @ai_function
    async def use_item(self,
        player_name: Annotated[str, AIParam(desc="The name of the player charater who wants to use the item from the inventory.")],
        item_name: Annotated[str, AIParam(desc="The name of the item which the player wants to use.")]
    ):
        """
        Let the player use an item if the player wants to use it from the inventory. 
        If the samples returned from `use_random_table` function are related to using an item by a player, call this function after sampling. 
        Do not call this function if the item does not exist in the inventory of the player who triggered this function.
        """
        arguments = {'player_name': player_name, 'item_name': item_name}

        # Wrong argument: The player does not exist.
        if player_name not in self.name_to_idx:
            msg = f"THE PLAYER NAME {player_name} CANNOT BE FOUND."
            print_system_log(msg, after_break=True)
            return msg, arguments, None

        player = self.players[self.name_to_idx[player_name]]

        # Wrong activation/argument: The item does not exist.
        if item_name not in player.inventory:
            msg = f"THE PLAYER {player_name} DOES NOT HAVE THE ITEM {item_name}."
            print_system_log(msg, after_break=True)
            return msg, arguments, None

        # The default system prompt consists of the instruction to check if the item is expendable.
        system_prompt = ' '.join(EXPENDABLE_CHECK_PROMPT)
        scene_prompt = self.make_scene_prompt()
        player_prompt = self.make_player_prompt(player)
        system_prompt = f"{system_prompt}\n\nScene State: {scene_prompt.content}\n\nPlayer State: {player_prompt.content}"
        
        options = ['Expendable', 'Not expendable']
        options_str = '\n'.join([f"{o}: {option}" for o, option in enumerate(options)])
        kani = Kani(self.engine, system_prompt=system_prompt)
        generation_params = {
            'temperature': 0.2,
            'top_p': 1,
            'presence_penalty': 0,
            'frequency_penalty': 0,
        }

        res = await kani.chat_round_str(f"Is the item expendable which should be removed after usage?\n\n{item_name}: {player.inventory[item_name]}\n\n{options_str}", **generation_params)
        res = convert_into_class_idx(res, options)

        intermediate_res = {f"The item '{item_name}' expendable": True if res == 0 else False}

        if res == 0:  # The item is expendable.
            msg = f"THE PLAYER {player_name} USED THE ITEM {item_name}. IT HAS BEEN REMOVED FROM THE INVENTORY SINCE IT IS AN EXPENDABLE ITEM."
            print_system_log(msg, after_break=True)
            player.remove_item(item_name)
            return msg, arguments, intermediate_res

        # The item is permanent.
        msg = f"THE PLAYER {player_name} USED THE ITEM {item_name}."
        print_system_log(msg, after_break=True)
        return msg, arguments, intermediate_res

    # Kani's function call for adding an object into the environment.
    @ai_function
    def add_object(self,
        object_name: Annotated[str, AIParam(desc="The name of the object which should be added into the environment.")],
        object_desc: Annotated[str, AIParam(desc="The description of the object.")]
    ):
        """
        Add a new object to the environment in the scene if any circumstance necessitates it. 
        Pass the description of the object as a parameter too if it exists. 
        If it does not exist, generate a brief description in one or two sentences. 
        If the samples returned from `use_random_table` function are related to setting a new object to the environment, call this function after sampling. 
        Do not call this function if the object already exists in the environment.
        """
        arguments = {'object_name': object_name, 'object_desc': object_desc}

        # Wrong activation/argument: The object already exists.
        if object_name in self.environment:
            msg = f"THE OBJECT {object_name} ALREADY EXISTS IN THE ENVIRONMENT."
            print_system_log(msg, after_break=True)
            return msg, arguments, None

        self.environment[object_name] = object_desc

        msg = f"A NEW OBJECT {object_name} HAS BEEN ADDED TO THE ENVIRONMENT IN THE CURRENT SCENE."
        print_system_log(msg, after_break=True)
        return msg, arguments, None

    # Kani's function call for getting access to an object in the environment.
    @ai_function
    async def use_environment(self, 
        player_name: Annotated[str, AIParam(desc="The name of the player charater who tries to reach out to an object in the environment.")], 
        object_name: Annotated[str, AIParam(desc="The name of the object in the environment to be accessed.")]
    ):
        """
        Let the player get access to an object or a location in the environment if the player tries to reach out to it anytime during the game. 
        If the samples returned from `use_random_table` function are related to interacting with an existing object in the environment, call this function after sampling. 
        Do not call this function if the object does not exist in the current environment.
        """

        arguments = {'player_name': player_name, 'object_name': object_name}

        # Wrong argument: The player does not exist.
        if player_name not in self.name_to_idx:
            msg = f"THE PLAYER NAME {player_name} CANNOT BE FOUND."
            print_system_log(msg, after_break=True)
            return msg, arguments, None

        player = self.players[self.name_to_idx[player_name]]

        # Wrong activation/argument: The object does not exist.
        if object_name not in self.environment:
            msg = f"THE OBJECT {object_name} CANNOT BE FOUND IN THE ENVIRONMENT."
            print_system_log(msg, after_break=True)
            return msg, arguments, None
        
        object_desc = self.environment[object_name]

        # The default system prompt consists of the instruction to check if the object is obtainable.
        system_prompt = ' '.join(OBTAINABLE_CHECK_PROMPT)
        scene_prompt = self.make_scene_prompt()
        system_prompt = f"{system_prompt}\n\nScene State: {scene_prompt.content}"
        
        options = ['Obtainable', 'Not obtainable']
        options_str = '\n'.join([f"{o}: {option}" for o, option in enumerate(options)])
        kani = Kani(self.engine, system_prompt=system_prompt)
        generation_params = {
            'temperature': 0.2,
            'top_p': 1,
            'presence_penalty': 0,
            'frequency_penalty': 0,
        }

        res = await kani.chat_round_str(f"Is this object obtainable which can be stored in the player inventory?\n\n{object_name}: {object_desc}\n\n{options_str}", **generation_params)
        res = convert_into_class_idx(res, options)

        intermediate_res = {
            f"The object '{object_name}' obtainable": True if res == 0 else False,
            f"The item '{object_name}' added": False
        }

        if res == 0:  # The item is obtainble.
            # Removing unnecessary punctuations from the object name.
            item_name = remove_punctuation(object_name)

            print_system_log(f"{item_name}: {object_desc}")
            print_system_log("ARE YOU GOING TO TAKE THIS ITEM?")
            selected = select_random_options(['Yes', 'No']) if isinstance(player, PlayerKani) else select_options(['Yes', 'No'])

            if selected == 0:
                obtain_msg = self.put_item(player, item_name, object_desc)

                # Checking if the player took the item to update the environment.
                if item_name in player.inventory:
                    self.environment.pop(object_name)

                intermediate_res[f"The item '{object_name}' added"] = True
                msg = f"THE PLAYER {player_name} FOUND THE ITEM {item_name}.\n{obtain_msg}"
                print_system_log(msg, after_break=True)

                return msg, arguments, intermediate_res
            
            msg = f"THE PLAYER {player_name} FOUND THE ITEM {item_name}, BUT DECIDED NOT TO TAKE IT."
            print_system_log(msg, after_break=True)

            return msg, arguments, intermediate_res

        msg = f"THE PLAYER {player_name} FOUND {object_name}. IT SEEMS NOT OBTAINABLE."
        print_system_log(msg, after_break=True)

        return msg, arguments, intermediate_res

    # Kani's function call for getting access to the random table.
    @ai_function
    async def use_random_table(self, 
        table_name: Annotated[str, AIParam(desc="The name of the table to be accessed.")]
    ):
        """
        Sample some entries from a random table when it should be referred to anytime during the game. 
        Do not call this function if the name of the required table does not exist in the random table dictionary. 
        Note that this function can be called flexibly when there should be a random encounter or the current scene should be updated with a new NPC or object during the game.
        """

        arguments = {'table_name': table_name}

        # Wrong activation/argument: The table name does not exist in the random table dictionary.
        if table_name not in self.random_tables:
            msg = f"THERE IS NO RANDOM TABLE {table_name}."
            print_system_log(msg, after_break=True)
            return msg, arguments, None

        entries = self.random_tables[table_name]

        system_prompt = ' '.join(TABLE_PROCESSING_PROMPT)
        scene_prompt = self.make_scene_prompt()
        system_prompt = f"{system_prompt}\n\nScene State: {scene_prompt.content}"
        kani = Kani(self.engine, chat_history=clean_history(self.current_queries), system_prompt=system_prompt)

        intermediate_res = {}

        # 1. Determining the number of samples to retrieve.
        generation_params = {
            'temperature': 0.2,
            'top_p': 1,
            'presence_penalty': 0,
            'frequency_penalty': 0,
        }

        query = "How many entries should be sampled from the table? If the specific number is indicated in the scene, you should give that number. " + \
            "If not, you can determine any number which you think most reasonable. " + \
            "You must answer only in number."
        res = await kani.chat_round_str(f"{query}\n\nTarget table: {table_name}", **generation_params)
        num_samples = convert_into_number(res)
        if num_samples is None: num_samples = random.randint(1, len(entries))
        intermediate_res["The number of samples"] = num_samples

        # 2. Sampling the entries.
        samples = random.sample(entries, num_samples)
        intermediate_res["The retrieved samples from the table"] = deepcopy(samples)

        # 3. Updating the table after sampling.
        exclusion_options = ['Yes', 'No']
        options_str = '\n'.join([f"{o}: {option}" for o, option in enumerate(exclusion_options)])
        query = "Do you think the sampled entries should be excluded from the table because they will not be needed later? " + \
            "You must answer only in number."
        res = await kani.chat_round_str(f"{query}\n\nTarget table: {table_name}\n\nRetrieved samples: {samples}\n\n{options_str}", **generation_params)
        exclusion_idx = convert_into_class_idx(res, exclusion_options)

        if exclusion_idx == 0:  # The retrieved sample should be excluded from the table.
            entries = [entry for entry in entries if entry not in samples]
        self.random_tables[table_name] = entries
        if len(entries) == 0:
            self.random_tables.pop(table_name)
        intermediate_res["Exclusion of the sampled entries"] = True if exclusion_idx == 0 else False

        # 4. Determining whether the random table should be removed or not.
        if table_name in self.random_tables:
            removal_options = ["Yes", "No"]
            options_str = '\n'.join([f"{o}: {option}" for o, option in enumerate(removal_options)])
            query = "Do you think the table should be removed because it will not be required anymore? " + \
                "You must answer only in number."
            res = await kani.chat_round_str(f"{query}\n\nTarget table: {table_name}\n\n{options_str}", **generation_params)
            removal_idx = convert_into_class_idx(res, removal_options)
            if removal_idx == 0:
                self.random_tables.pop(table_name)
            intermediate_res["Removal of the table"] = True if removal_idx == 0 else False

        samples_str = '\n'.join(samples)
        msg = f"SAMPLED FROM THE TABLE {table_name}: \n{samples_str}\n\nRUN ANOTHER FUNCTION IF THE RESULT REQUIRES TO ADD OR CHANGE ANY OBJECTS OR NPCS IN THE SCENE."
        print_system_log(msg, after_break=True)
        return msg, arguments, intermediate_res

    # Validating if the current interaction falls into the success condition.
    async def validate_success_condition(self):
        if len(self.success_condition) == 0:
            return False

        # The default system prompt consists of the instruction and the success condition.
        system_prompt = ' '.join(VALIDATE_SUCCESS_PROMPT)

        options = ['Succeeded', 'Not yet']
        options_str = '\n'.join([f"{o}: {option}" for o, option in enumerate(options)])
        kani = Kani(self.engine, chat_history=deepcopy(self.raw_history), system_prompt=system_prompt)
        generation_params = {
            'temperature': 0.2,
            'top_p': 1,
            'presence_penalty': 0,
            'frequency_penalty': 0,
        }

        res = await kani.chat_round_str(f"Have the players accomplished the success condition?\n\nSuccess condition: {self.success_condition}\n\n{options_str}", **generation_params)
        res = convert_into_class_idx(res, options)

        return True if res == 0 else False
    
    # Validating if the current interaction falls into the failure condition.
    async def validate_failure_condition(self):
        if len(self.failure_condition) == 0:
            return False

        # The default system prompt consists of the instruction and the failure condition.
        system_prompt = ' '.join(VALIDATE_FAILURE_PROMPT)

        options = ['Failed', 'Not yet']
        options_str = '\n'.join([f"{o}: {option}" for o, option in enumerate(options)])
        kani = Kani(self.engine, chat_history=deepcopy(self.raw_history), system_prompt=system_prompt)
        generation_params = {
            'temperature': 0.2,
            'top_p': 1,
            'presence_penalty': 0,
            'frequency_penalty': 0,
        }

        res = await kani.chat_round_str(f"Have the players fallen into the failure condition?\n\nFailure condition: {self.failure_condition}\n\n{options_str}", **generation_params)
        res = convert_into_class_idx(res, options)
        
        return True if res == 0 else False
