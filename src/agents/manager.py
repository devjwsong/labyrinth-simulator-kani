from kani import Kani, ai_function, AIParam
from kani.models import ChatMessage, ChatRole, FunctionCall, QueryType
from kani.exceptions import FunctionCallException, MessageTooLong, NoSuchFunction, WrappedCallException
from kani.internal import FunctionCallResult
from kani.utils.message_formatters import assistant_message_contents
from kani.engines.base import BaseCompletion
from sentence_transformers import SentenceTransformer, util
from agents.player import Player
from constants import (
    SEP,
    RULE_SUMMARY,
    VALIDATE_SUCCESS_PROMPT, 
    VALIDATE_FAILURE_PROMPT,
    DIFFICULTY_PROMPT,
    CREATE_NPC_PROMPT,
    OBTAINABLE_CHECK_PROMPT,
    EXPENDABLE_CHECK_PROMPT, 
    SUMMARIZE_PROMPT,
    GENERATE_TRAIT_DESC_PROMPT,
    GENERATE_FLAW_DESC_PROMPT,
    GENERATE_OBJECT_DESC_PROMPT
)
from utils import (
    print_system_log, 
    remove_punctuation, 
    select_options, 
    select_random_options, 
    find_current_point, 
    convert_into_dict,
    convert_into_natural, 
    convert_into_class_idx
)
from typing import AsyncIterable, Annotated, Tuple, Callable
from argparse import Namespace
from copy import deepcopy
from itertools import chain

import json
import logging
import random
import numpy as np
import torch

log = logging.getLogger("kani")
message_log = logging.getLogger("kani.messages")


# The whole game manager class.
class GameManager(Kani):
    def __init__(self, scene: dict, main_args: Namespace, encoder: SentenceTransformer, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Attributes which should be initialized before the game.
        self.chapter = scene['chapter']
        self.scene = scene['scene']
        self.scene_summary = scene['scene_summary']
        self.npcs = scene['npcs']
        self.generation_rules = scene['generation_rules']
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

        # Context prompts.
        self.rule_prompt = None
        self.scene_prompt = None
        self.player_prompts = []

        # Additional attributes for enabling the prompt policies.
        self.encoder = encoder
        self.sent_embs = np.empty((0, encoder.get_sentence_embedding_dimension())) if self.concat_policy == 'retrieval' else None
        self.raw_history = []
        self.start_idx = 0
        self.turn_count = 0
        self.context_archive = []
        self.function_arguments = {}
        self.function_intermediate_res = {}

        # Additional attributes for game play.
        self.players = []
        self.name_to_idx = {}
        self.automated_player = main_args.automated_player
        self.is_action_scene = False

        # Pre-buidling the rule prompt or embeddings.
        self.game_rules = []
        self.rule_embs = None
        if main_args.rule_injection == 'full':
            rule_content = '\n'.join([' '.join(part) for part in RULE_SUMMARY])
            self.rule_prompt = ChatMessage.system(name="Game_Rules", content=rule_content)
        elif main_args.rule_injection == 'retrieval':
            self.game_rules = list(chain.from_iterable(RULE_SUMMARY))
            self.rule_embs = np.empty((0, encoder.get_sentence_embedding_dimension()))
            for sent in self.game_rules:
                emb = self.encoder.encode(sent)
                self.rule_embs = np.concatenate((self.rule_embs, np.expand_dims(emb, axis=0)))

            assert self.rule_embs.shape[0] == len(self.game_rules), "The number of rule embeddings should be identical to the length of rule list."

    # Setting the attributes in the scene.
    def set_scene(self, obj):
        self.chapter = obj['chapter']
        self.scene = obj['scene']
        self.scene_summary = obj['scene_summary']
        self.npcs = obj['npcs']
        self.generation_rules = obj['generation_rules']
        self.success_condition = obj['success_condition']
        self.failure_condition = obj['failure_condition']
        self.game_flow = obj['game_flow']
        self.environment = obj['environment']
        self.random_tables = obj['random_tables']
        self.consequences = obj['consequences']

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

    # Getter for generation rules in a (numbered) list format.
    def get_generation_rules(self, with_number=False):
        if with_number:
            return [f"({r+1}) {rule}" for r, rule in enumerate(self.generation_rules)]
        return self.generation_rules

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

        print("<GENERATION RULES>")
        print('\n'.join(self.get_generation_rules(with_number=True)))

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

    # Encoding the chat message into a sentence embedding.
    def encode_message(self, message: ChatMessage):
        content = convert_into_natural(message)
        return self.encoder.encode(content)

    # Overriding add_to_history.
    async def add_to_history(self, message: ChatMessage, store_in_raw: bool=True):
        self.chat_history.append(message)
        if store_in_raw:
            self.raw_history.append(message)

        # Sentence embedding for the retrieval.
        if self.sent_embs is not None:
            emb = self.encode_message(message)
            self.sent_embs = np.concatenate((self.sent_embs, np.expand_dims(emb, axis=0)))

            # The number of sentence embeddings and chat logs should always be identical.
            assert len(self.chat_history) == self.sent_embs.shape[0], "The sentence embeddings and chat histories are not synced."

    # Making a prompt using the simple concatenation.
    def get_simple_history(self) -> list[ChatMessage]:
        if self.max_num_msgs is None:
            valid_chat_history = deepcopy(self.chat_history)
        else:
            valid_chat_history = self.chat_history[max(len(self.chat_history)-self.max_num_msgs, 0):]

        return valid_chat_history

    # Making a prompt using the retrieval concatenation.
    def get_retrieval_history(self) -> list[ChatMessage]:
        cur = find_current_point(self.chat_history)

        # If this is the case, retrieval has no meaning.
        if len(self.chat_history) <= self.max_num_msgs or self.max_num_msgs <= (len(self.chat_history)-cur):
            return self.get_simple_history()
        
        # Calculating the max-pooled cosine similarities.
        top_n = self.max_num_msgs - (len(self.chat_history)-cur)
        query_embs, cand_embs = self.sent_embs[cur:], self.sent_embs[:cur]  # (Q, d), (C, d)
        cos_sims = util.cos_sim(query_embs, cand_embs)  # (Q, C)
        scores = torch.max(cos_sims, dim=0).values  # (C)

        # Sorting the candidate logs by the similarities.
        idxs = torch.sort(scores, descending=True).indices[:top_n]
        idxs = torch.sort(idxs).values
        valid_chat_history = [self.chat_history[:cur][idx] for idx in idxs]
        valid_chat_history += self.chat_history[cur:]

        # Checking the length of the valid chat logs.
        assert len(valid_chat_history) == self.max_num_msgs, "The numbers of sampled chat logs and the maximum number of turns are different."

        return valid_chat_history

    # Summarizing the given dialogue history.
    async def summarize_history(self, input_history: list[ChatMessage]) -> ChatMessage:
        # The default system prompt for the instruction.
        system_prompt = ' '.join(SUMMARIZE_PROMPT)
        
        kani = Kani(self.engine, chat_history=input_history, system_prompt=system_prompt)
        res = await kani.chat_round_str("Give me the summarization of the chat history so far.")

        return ChatMessage.system(content=res, name="Summary")

    # Overriding get_prompt.
    async def get_prompt(self) -> list[ChatMessage]:
        # First, setting the additional information.
        self.make_rule_prompt()
        self.make_scene_prompt()
        self.make_player_prompts()

        rule_prompt_len = 0
        if self.rule_prompt is not None:
            rule_prompt_len = self.message_token_len(self.rule_prompt)
        scene_prompt_len = 0
        if self.scene_prompt is not None:
            scene_prompt_len = self.message_token_len(self.scene_prompt)
        always_len = self.always_len + rule_prompt_len + scene_prompt_len  # Additional length for rule/scene information.
        for message in self.player_prompts:  # Additional length for player information.
            always_len += self.message_token_len(message)

        # If summarization + no period, valid_chat_history is just one summary and the current query.
        if self.summarization and self.summ_period is None:
            cur = find_current_point(self.chat_history)
            summary = await self.summarize_history(self.chat_history[:cur])
            valid_chat_history = [summary] + self.chat_history[cur:]
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

        default_prompt = deepcopy(self.always_included_messages)
        if self.rule_prompt is not None:
            default_prompt += [self.rule_prompt]
        if self.scene_prompt is not None:
            default_prompt += [self.scene_prompt]
        if len(self.player_prompts) > 0:
            default_prompt += self.player_prompts

        if not to_keep:
            return default_prompt
        prompt = default_prompt + valid_chat_history[-to_keep:]

        return prompt

    # Making the rule prompt.
    def make_rule_prompt(self, top_n: int=5):
        # If this is a full rule injection or none rull injection, return None.
        if self.rule_embs is None:
            return

        # Calculating the cosine similarities between the queries and rules.
        cur = find_current_point(self.chat_history)
        if self.sent_embs is not None:
            query_embs = self.sent_embs[cur:]  # (Q, d)
        else:
            query_embs = np.empty((0, self.encoder.get_sentence_embedding_dimension()))
            queries = self.chat_history[cur:]
            for query in queries:
                emb = self.encode_message(query)
                query_embs = np.concatenate((query_embs, np.expand_dims(emb, axis=0)))
        cos_sims = util.cos_sim(query_embs, self.rule_embs)  # (Q, C)
        scores = torch.max(cos_sims, dim=0).values  # (C)

        # Sorting the candidate logs by the similarities.
        idxs = torch.sort(scores, descending=True).indices[:top_n]
        idxs = torch.sort(idxs).values
        valid_rules = [self.game_rules[idx] for idx in idxs]

        rule_content = '\n'.join(valid_rules)
        self.rule_prompt = ChatMessage.system(name="Game_Rules", content=rule_content)

    # Making the scene prompt.
    def make_scene_prompt(self):
        # If there is no scene, return None.
        if len(self.chapter) == 0 and len(self.scene) == 0: return

        content = f"chapter={self.chapter}, scene={self.scene}, scene_summary={self.scene_summary}, " + \
            f"npcs={self.npcs}, generation_rules={self.generation_rules}, success_condition={self.success_condition}, failure_condition={self.failure_condition}, " + \
            f"game_flow={self.game_flow}, environement={self.environment}, random_tables={self.random_tables}, consequences={self.consequences}, " + \
            f"is_action_scene={self.is_action_scene}"
        self.scene_prompt = ChatMessage.system(name="Scene_State", content=content)

    # Making one player prompt.
    def make_player_prompt(self, player: Player):
        content = f"name={player.name}, kin={player.kin}, persona={player.persona}, goal={player.goal}, " + \
            f"traits={player.traits}, flaws={player.flaws}, inventory={player.inventory}, additional_notes={player.additional_notes}"
        return ChatMessage.system(name="Player_State", content=content)

    # Making the player prompts.
    def make_player_prompts(self):
        # If there is no player, return None.
        if len(self.players) == 0: return

        self.player_prompts.clear()
        for player in self.players:
            self.player_prompts.append(self.make_player_prompt(player))

    # Making the context for exporting data.
    def make_context(self):
        context = {
            "scene": {
                "chapter": self.chapter,
                "scene": self.scene,
                "scene_summary": deepcopy(self.scene_summary),
                "npcs": deepcopy(self.npcs),
                "generation_rules": deepcopy(self.generation_rules),
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
    async def get_model_completion(self, include_functions: bool = True, **kwargs) -> Tuple[BaseCompletion, list[ChatMessage]]:
        """Get the model's completion with the current chat state.

        Compared to :meth:`chat_round` and :meth:`full_round`, this lower-level method does not save the model's reply
        to the chat history or mutate the chat state; it is intended to help with logging or to repeat a call multiple
        times.

        :param include_functions: Whether to pass this kani's function definitions to the engine.
        :param kwargs: Arguments to pass to the model engine.
        """
        # get the current chat state
        messages = await self.get_prompt()

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
        past_history = deepcopy(self.raw_history)
        current_queries = deepcopy(queries)

        for msg in queries:
            await self.add_to_history(msg)

        retry = 0
        is_model_turn = True
        async with self.lock:
            while is_model_turn:
                # Making the context for exporting data.
                context = self.make_context()
                context["past_history"] = []
                for msg in past_history:
                    context["past_history"].append(convert_into_dict(msg))
                context["current_queries"] = []
                for msg in current_queries:
                    context["current_queries"].append(convert_into_dict(msg))

                # do the model prediction
                completion, messages = await self.get_model_completion(**kwargs)
                context["actual_prompt"] = []
                for msg in messages:
                    context["actual_prompt"].append(convert_into_natural(msg))

                message = completion.message
                if message.content is not None:  # Ignoring pending function calling.
                    if message.role == ChatRole.ASSISTANT:
                        message = ChatMessage.assistant(name="Goblin_King", content=message.content)
                    await self.add_to_history(message)
                yield message

                # if function call, do it and attempt retry if it's wrong
                if not message.function_call:
                    context["generated"] = convert_into_dict(message)
                    self.context_archive.append(context)
                    break

                try:
                    func_res = await self.do_function_call(message.function_call)
                    is_model_turn = func_res.is_model_turn
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

                context["generated"] = convert_into_dict(func_res.message)
                
                # If the generated message is from a function, the arguments and intermediate results should be stored.
                context["function_arguments"] = deepcopy(self.function_arguments)
                context["function_intermediate_results"] = deepcopy(self.function_intermediate_res)
                self.function_arguments.clear()
                self.function_intermediate_res.clear()

                self.context_archive.append(context)
                current_queries.append(func_res.message)

            # Increasing the turn count. If the summarization period has been reached, adding the summary.
            self.turn_count += 1
            if self.summarization and self.summ_period is not None and self.turn_count == self.summ_period:
                input_history = self.chat_history[self.start_idx:]
                summary = await self.summarize_history(input_history)
                await self.add_to_history(summary, store_in_raw=False)

                if self.clear_raw_logs:
                    self.chat_history = self.chat_history[:self.start_idx] + self.chat_history[-1:]
                    
                    if self.sent_embs is not None:
                        self.sent_embs = np.concatenate((self.sent_embs[:self.start_idx], self.sent_embs[-1:]), axis=0)
                
                self.start_idx = len(self.chat_history)
                self.turn_count = 0

    # Overriding do_function_call.
    async def do_function_call(self, call: FunctionCall) -> FunctionCallResult:
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

        # Adding the function result in the chat history.
        await self.add_to_history(msg)

        return FunctionCallResult(is_model_turn=f.after == ChatRole.ASSISTANT, message=msg)

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
                yield text, message.role

    # Overriding chat_round.
    async def chat_round(self, query: QueryType, **kwargs) -> ChatMessage:
        """Perform a single chat round (user -> model -> user, no functions allowed).

        This is slightly faster when you are chatting with a kani with no AI functions defined.

        :param query: The contents of the user's chat message.
        :param kwargs: Additional arguments to pass to the model engine (e.g. hyperparameters).
        :returns: The model's reply.
        """
        kwargs = {**kwargs, "include_functions": False}
        # do the chat round
        async with self.lock:
            # add the user's chat input to the state
            await self.add_to_history(ChatMessage.user(query))

            # and get a completion
            completion, _ = await self.get_model_completion(**kwargs)
            message = completion.message
            await self.add_to_history(message)
            return message

    # Kani's function call for a dice roll test.
    @ai_function
    async def activate_test(self, 
        player_name: Annotated[str, AIParam(desc="The name of the player charater who should do the test.")],
        initial_difficulty: Annotated[int, AIParam(desc="The initially set difficulty of the task in a range of 2 and 6.")],
        final_difficulty: Annotated[int, AIParam(desc="The final difficulty which has been reduced by the teamwork of the party and the minimum should still be 2.")]
    ):
        """
        Activate a test if a player is trying to do something with a certain difficulty.
        Determine the original difficulty of the task first and then set the final difficulty after reducing it depending on the teamwork from other players.
        """

        player = self.players[self.name_to_idx[player_name]]

        # The default system prompt consists of the instruction.
        system_prompt = ' '.join(DIFFICULTY_PROMPT)
        
        options = ["The test becomes easier.", "The test becomes harder.", "There is no change."]
        options_str = '\n'.join([f"{o}: {option}" for o, option in enumerate(options)])
        kani = Kani(self.engine, chat_history=[self.make_player_prompt(player)] + self.chat_history, system_prompt=system_prompt)
        res = await kani.chat_round_str(f"Would the test become easier, harder, or none of them depending on the player traits or flaws?\n\n{options_str}")
        res = convert_into_class_idx(res, options)

        if res == 2:  # The difficulty is not affected.
            if not self.automated_player:
                _ = input(f"THE TEST DIFFICULTY: {final_difficulty}: PRESS ANY KEY TO ROLL A DICE.")
            dice_result = random.randint(1, 6)

        elif res == 0:  # The test is improved.
            print_system_log("A TRAIT IN THE PLAYER MAKES THE TEST EASIER. YOU ROLL TWO DICES AND TAKE THE LARGER ONE.")
            if not self.automated_player:
                _ = input(f"THE TEST DIFFICULTY: {final_difficulty}: PRESS ANY KEY TO ROLL TWO DICES.")
            result1, result2 = random.randint(1, 6), random.randint(1, 6)
            dice_result = max(result1, result2)
            print_system_log(f"RESULT 1 ({result1}) vs RESULT 2 ({result2}) => THE PLAYER GOT {dice_result}.")

        elif res == 1:  # The test is hindered.
            print_system_log("A FLAW IN THE PLAYER MAKES THE TEST HARDER. YOU ROLL TWO DICES AND TAKE THE SMALLER ONE.")
            if not self.automated_player:
                _ = input(f"THE TEST DIFFICULTY: {final_difficulty}: PRESS ANY KEY TO ROLL TWO DICES.")
            result1, result2 = random.randint(1, 6), random.randint(1, 6)
            dice_result = min(result1, result2)
            print_system_log(f"RESULT 1 ({result1}) vs RESULT 2 ({result2}) => THE PLAYER GOT {dice_result}.")

        if dice_result < final_difficulty:
            msg = f"TEST FAILED. THE DICE ROLL RESULT {dice_result} IS SMALLER THAN THE DIFFICULTY {final_difficulty}."
        else:
            msg = f"TEST SUCCEEDED. THE DICE ROLL RESULT {dice_result} IS LARGER THAN OR EQUAL TO THE DIFFICULTY {final_difficulty}."
        
        self.function_arguments['player_name'] = player_name
        self.function_arguments['initial_difficulty'] = initial_difficulty
        self.function_arguments['final_difficulty'] = final_difficulty
        self.function_intermediate_res[f"Improvement/Hinderance of the test due to the player traits/flaws"] = options[res]

        # Updating the new chat message.
        print_system_log(msg, after_break=True)
        return msg

    # Kani's function call for starting an action scene.
    @ai_function
    def activate_action_scene(self):
        """Activate an action scene if this is a circumstance that players should take actions in a tight time limit."""

        # False Positive: The function is called even when the action scene has already been activated.
        if self.is_action_scene:
            msg = "UNEXPECTED FUNCTION CALLING: THE ACTION SCENE HAS ALREADY BEEN ACTIVATED."
            print_system_log(msg, after_break=True)
            return msg

        self.is_action_scene = True
        msg = "ACTION SCENE ACTIVATED."
        print_system_log(msg, after_break=True)
        return msg

    # Kani's function call for ending an action scene.
    @ai_function
    def terminate_action_scene(self):
        """Terminate the current ongoing action scene if the urgent circumstance has been finished now."""

        # False Positive: The function is called even when the action scene has not been activated before.
        if not self.is_action_scene:
            msg = "UNEXPECTED FUNCTION CALLING: THE ACTION SCENE HAS NOT BEEN ACTIVATED YET."
            print_system_log(msg, after_break=True)
            return msg

        self.is_action_scene = False
        msg = "ACTION SCENE TERMINATED."
        print_system_log(msg, after_break=True)
        return msg

    # Kani's function call for creating an NPC immediately.
    @ai_function
    async def create_npc(self, npc_name: Annotated[str, AIParam(desc="The name of the NPC which has been requested by the player.")]):
        """
        Create an NPC if the NPC requested by a user does not exist in the scene yet.
        This function must not be called if the NPC already exists in the scene.
        """ 

        # False Positive: The function is called even when the argument is an NPC which already exists.
        if npc_name in self.npcs:
            msg = "UNEXPECTED FUNCTION CALLING: NPC ALREADY EXISTS."
            print_system_log(msg, after_break=True)
            return msg

        # The default system prompt consists of the instruction and the requirement for an NPC.
        system_prompt = ' '.join(CREATE_NPC_PROMPT)
        
        kani = Kani(self.engine, chat_history=[self.scene_prompt], system_prompt=system_prompt)
        res = await kani.chat_round_str(f"Generate the specifications of the requested NPC.\nNPC name: '{npc_name}'")

        # Converting & Fetching information.
        try:
            res = json.loads(res)

            assert isinstance(res['kin'], str), "THE KIN OF AN NPC IS NOT THE STRING TYPE."
            assert isinstance(res['persona'], list), "THE PERSONA OF AN NPC IS NOT THE LIST TYPE."
            assert isinstance(res['goal'], str), "THE GOAL OF AN NPC IS NOT THE STRING TYPE."
            assert isinstance(res['trait'], str), "THE TRAITS OF AN NPC IS NOT THE STRING TYPE."
            assert isinstance(res['flaw'], str), "THE FLAWS OF AN NPC IS NOT THE STRING TYPE."

            self.npcs[npc_name] = res

        except json.decoder.JSONDecodeError as e:
            log.debug(res)
            log.error(f"{e}: The output format cannot be converted into dict.")
            raise Exception()
        except KeyError as e:
            log.debug(res)
            log.error(f"{e}: Missing key.")
            raise Exception()

        self.function_arguments['npc_name'] = npc_name
        self.function_intermediate_res[f"Generated information of the NPC '{npc_name}'"] = res

        msg = f"NPC {npc_name} CREATED."
        print_system_log(msg, after_break=True)
        return msg

    # Kani's function call for adding a trait to the player.
    @ai_function
    async def add_trait(self,
        player_name: Annotated[str, AIParam(desc="The name of the player charater whose new trait should be added.")],
        trait_name: Annotated[str, AIParam(desc="The name of the new trait to be added in the player.")]
    ):
        """
        Add a new trait to a player if any circumstance necessiates it.
        """

        player = self.players[self.name_to_idx[player_name]]
        
        # The default system prompt consists of the instruction to generate the specific description of the trait.
        system_prompt = ' '.join(GENERATE_TRAIT_DESC_PROMPT)
        kani = Kani(self.engine, chat_history=[self.make_player_prompt(player)], system_prompt=system_prompt)
        trait_desc = await kani.chat_round_str(f"Generate the plausible description of the trait.\nTrait: {trait_name}")

        player.add_trait(trait_name, trait_desc)
        
        self.function_arguments['player_name'] = player_name
        self.function_arguments['trait_name'] = trait_name
        self.function_intermediate_res[f"Generated description of the trait '{trait_name}'"] = trait_desc

        msg = f"A NEW TRAIT {trait_name}: {trait_desc} HAS BEEN ADDED TO THE PLAYER {player_name}."
        updated_res = '\n'.join(player.get_traits(with_number=True))
        print_system_log(f"PLAYER TRAITS UPDATED:\n{updated_res}", after_break=True)
        print_system_log(msg, after_break=True)
        return msg

    # Kani's function call for adding a flaw to the player.
    @ai_function
    async def add_flaw(self,
        player_name: Annotated[str, AIParam(desc="The name of the player charater whose new flaw should be added.")],
        flaw_name: Annotated[str, AIParam(desc="The name of the new flaw to be added in the player.")]
    ):
        """
        Add a new flaw to a player if any circumstance necessiates it.
        """

        player = self.players[self.name_to_idx[player_name]]
        
        # The default system prompt consists of the instruction to generate the specific description of the trait.
        system_prompt = ' '.join(GENERATE_FLAW_DESC_PROMPT)
        kani = Kani(self.engine, chat_history=[self.make_player_prompt(player)], system_prompt=system_prompt)
        flaw_desc = await kani.chat_round_str(f"Generate the plausible description of the flaw.\nFlaw: {flaw_name}")

        player.add_flaw(flaw_name, flaw_desc)

        self.function_arguments['player_name'] = player_name
        self.function_arguments['flaw_name'] = flaw_name
        self.function_intermediate_res[f"Generated description of the flaw '{flaw_name}'"] = flaw_desc

        msg = f"A NEW FLAW {flaw_name}: {flaw_desc} HAS BEEN ADDED TO THE PLAYER {player_name}."
        updated_res = '\n'.join(player.get_flaws(with_number=True))
        print_system_log(f"PLAYER FLAWS UPDATED:\n{updated_res}", after_break=True)
        print_system_log(msg, after_break=True)
        return msg

    # Logic for adding an item. (Not an AI function!)
    def add_item(self, player: Player, item_name: str, item_desc: str):
        # A sub logic that adds the item.
        def sub_logic(player, item_name, item_desc):
            player.add_item(item_name, item_desc)
            msg = f"THE PLAYER {player.name} ADDED THE ITEM {item_name} TO THE INVENTORY."
            updated_res = '\n'.join(player.get_inventory(with_number=True))
            print_system_log(f"PLAYER INVENTORY UPDATED:\n{updated_res}", after_break=True)
            return msg

        if len(player.inventory) >= 6:  # The player inventory is already full.
            print_system_log("YOUR INVENTORY IS ALREADY FULL. CHOOSE ONE ITEM TO DISCARD FROM THE INVENTORY OR DECIDE NOT TO TAKE THE CURRENT ITEM.")
            options = [
                "Discarding one item from the inventory.",
                "Not taking the found item."
            ]
            selected = select_random_options(options) if self.automated_player else select_options(options)
            if selected == 0:  # Discarding any item from the inventory.
                print_system_log("WHICH ITEM ARE YOU GOING TO DISCARD?")
                selected = select_random_options(player.get_inventory) if self.automated_player else select_options(player.get_inventory())
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
        Remove a trait if any circumstance necessiates it.
        This function must not be called if the trait does not exist in the current state of the player who triggered this function.
        """

        player = self.players[self.name_to_idx[player_name]]

        # False Positive: The function is called even when the argument is a trait which does not exist in the player.
        if trait_name not in player.traits:
            msg = f"UNEXPECTED FUNCTION CALLING: THE PLAYER {player_name} DOES NOT HAVE THE TRAIT {trait_name}."
            print_system_log(msg, after_break=True)
            return msg

        # Removing the trait from the player.
        player.remove_trait(trait_name)

        self.function_arguments['player_name'] = player_name
        self.function_arguments['trait_name'] = trait_name

        msg = f"THE TRAIT {trait_name} HAS BEEN REMOVED FROM THE PLAYER {player_name}."
        updated_res = '\n'.join(player.get_traits(with_number=True))
        print_system_log(f"PLAYER TRAITS UPDATED:\n{updated_res}", after_break=True)
        print_system_log(msg, after_break=True)

        return msg

    # Kani's function call for removing a flaw from the player.
    @ai_function
    def remove_flaw(self,
        player_name: Annotated[str, AIParam(desc="The name of the player charater whose flaw should be removed.")],
        flaw_name: Annotated[str, AIParam(desc="The name of the flaw which should be removed from the player.")]
    ):
        """
        Remove a flaw if any circumstance necessiates it.
        This function must not be called if the flaw does not exist in the current state of the player who triggered this function.
        """

        player = self.players[self.name_to_idx[player_name]]

        # False Positive: The function is called even when the argument is a flaw which does not exist in the player.
        if flaw_name not in player.flaws:
            msg = f"UNEXPECTED FUNCTION CALLING: THE PLAYER {player_name} DOES NOT HAVE THE FLAW {flaw_name}."
            print_system_log(msg, after_break=True)
            return msg

        # Removing the flaw from the player.
        player.remove_flaw(flaw_name)

        self.function_arguments['player_name'] = player_name
        self.function_arguments['flaw_name'] = flaw_name

        msg = f"THE FLAW {flaw_name} HAS BEEN REMOVED FROM THE PLAYER {player_name}."
        updated_res = '\n'.join(player.get_flaws(with_number=True))
        print_system_log(f"PLAYER FLAWS UPDATED:\n{updated_res}", after_break=True)
        print_system_log(msg, after_break=True)

        return msg

    # Kani's function call for removing an item in the player's inventory.
    @ai_function
    def remove_item(self,
        player_name: Annotated[str, AIParam(desc="The name of the player charater who wants to remove the item from the inventory.")],
        item_name: Annotated[str, AIParam(desc="The name of the item which the player wants to discard.")]
    ):
        """
        Remove an item if the player wants to discard it from the inventory.
        This function must not be called if the item does not exist in the inventory of the player who triggered this function.
        """

        player = self.players[self.name_to_idx[player_name]]

        # False Positive: The function is called even when the argument is an item which does not exist in the inventory.
        if item_name not in player.inventory:
            msg = f"UNEXPECTED FUNCTION CALLING: THE PLAYER {player_name} DOES NOT HAVE THE ITEM {item_name}."
            print_system_log(msg, after_break=True)
            return msg

        # Removing the item from the inventory.
        desc = player.inventory[item_name]
        player.remove_item(item_name)

        # Discarded item is placed in the environment.
        self.environment[item_name] = desc

        self.function_arguments['player_name'] = player_name
        self.function_arguments['item_name'] = item_name

        msg = f"THE PLAYER {player_name} REMOVED THE ITEM {item_name} FROM THE INVENTORY."
        updated_res = '\n'.join(player.get_inventory(with_number=True))
        print_system_log(f"PLAYER INVENTORY UPDATED:\n{updated_res}", after_break=True)
        print_system_log(msg, after_break=True)

        return msg

    # Kani's function call for using an item.
    @ai_function
    async def use_item(self,
        player_name: Annotated[str, AIParam(desc="The name of the player charater who wants to use the item from the inventory.")],
        item_name: Annotated[str, AIParam(desc="The name of the item which the player wants to use.")]
    ):
        """
        Let the player use an item if the player wants to use it from the inventory.
        This function must not be called if the item does not exist in the inventory of the player who triggered this function.
        """

        player = self.players[self.name_to_idx[player_name]]

        # False Positive: The function is called even when the argument is an item which does not exist in the inventory.
        if item_name not in player.inventory:
            msg = f"UNEXPECTED FUNCTION CALLING: THE PLAYER {player_name} DOES NOT HAVE THE ITEM {item_name}."
            print_system_log(msg, after_break=True)
            return msg

        # The default system prompt consists of the instruction to check if the item is expendable.
        system_prompt = ' '.join(EXPENDABLE_CHECK_PROMPT)
        
        options = ['Expendable', 'Permanent']
        options_str = '\n'.join([f"{o}: {option}" for o, option in enumerate(options)])
        kani = Kani(self.engine, chat_history=[self.scene_prompt, self.make_player_prompt(player)], system_prompt=system_prompt)
        res = await kani.chat_round_str(f"Is the item expendable which should be removed after usage?\n{item_name}: {player.inventory[item_name]}\n\n{options_str}")
        res = convert_into_class_idx(res, options)

        if res == 0:  # The item is expendable.
            msg = f"THE PLAYER {player_name} USED THE ITEM {item_name}. IT HAS BEEN REMOVED FROM THE INVENTORY SINCE IT IS AN EXPENDABLE ITEM."
            print_system_log(msg, after_break=True)
            player.remove_item(item_name)
            return msg

        self.function_arguments['player_name'] = player_name
        self.function_arguments['item_name'] = item_name
        self.function_intermediate_res[f"Expendable item detection result for '{item_name}'"] = options[res]

        # The item is permanent.
        msg = f"THE PLAYER {player_name} USED THE ITEM {item_name}."
        print_system_log(msg, after_break=True)
        return msg

    # Kani's function call for getting access to an object in the environment.
    @ai_function
    async def use_environment(self, 
        player_name: Annotated[str, AIParam(desc="The name of the player charater who tries to reach out to an object in the environment.")], 
        object_name: Annotated[str, AIParam(desc="The name of the object in the environment to be accessed.")]
    ):
        """
        Let the player get access to an object in the environment if the player tries to reach out to it or if the object should be referred to anytime during the game.
        This function must not be called if the object does not exist in the environment.
        If the object name also exists as a random table, ignore this function and call use_random_table function instead.
        """

        # False Positive: The function is called even when the argument is an object which does not exist in the environment.
        if object_name not in self.environment:
            msg = f"UNEXPECTED FUNCTION CALLING: THE OBJECT {object_name} DOES NOT EXIST IN THE ENVIRONMENT."
            print_system_log(msg, after_break=True)
            return msg
        
        object_desc = self.environment[object_name]

        # The default system prompt consists of the instruction to check if the object is obtainable.
        system_prompt = ' '.join(OBTAINABLE_CHECK_PROMPT)
        
        options = ['Obtainable', 'Not obtainable']
        options_str = '\n'.join([f"{o}: {option}" for o, option in enumerate(options)])
        kani = Kani(self.engine, chat_history=[self.scene_prompt], system_prompt=system_prompt)
        res = await kani.chat_round_str(f"Is this object obtainable which can be stored in the player inventory?\n{object_name}: {object_desc}\n\n{options_str}")
        res = convert_into_class_idx(res, options)

        self.function_arguments['player_name'] = player_name
        self.function_arguments['object_name'] = object_name
        self.function_intermediate_res[f"Obtainable object detection result for '{object_name}'"] = options[res]

        if res == 0:  # The item is obtainble.
            # Removing unnecessary punctuations from the object name.
            item_name = remove_punctuation(object_name)

            print_system_log(f"{item_name}: {object_desc}")
            print_system_log("ARE YOU GOING TO TAKE THIS ITEM?")
            selected = select_random_options(['Yes', 'No']) if self.automated_player else select_options(['Yes', 'No'])

            player_idx = self.name_to_idx[player_name]
            player = self.players[player_idx]

            if selected == 0:
                obtain_msg = self.add_item(player, item_name, object_desc)

                # Checking if the player took the item to update the environment.
                if item_name in player.inventory:
                    self.environment.pop(object_name)

                msg = f"THE PLAYER {player_name} FOUND THE ITEM {item_name}.\n{obtain_msg}"
                print_system_log(msg, after_break=True)

                return msg
            
            msg = f"THE PLAYER {player_name} FOUND THE ITEM {item_name}, BUT DECIDED NOT TO TAKE IT."
            print_system_log(msg, after_break=True)

            return msg

        msg = f"THE PLAYER {player_name} FOUND {object_name}. IT SEEMS NOT OBTAINABLE."
        print_system_log(msg, after_break=True)

        return msg

    # Kani's function call for getting access to the random table.
    @ai_function
    async def use_random_table(self, 
        player_name: Annotated[str, AIParam(desc="The name of the player charater which tries to reach out to the random table.")], 
        table_name: Annotated[str, AIParam(desc="The name of the table to be accessed.")]
    ):
        """
        Let the player use to a random table if the player tries to reach out to any random table or if a certain table should be referred to anytime during the game.
        This function must not be called if the table does not exist in the random table dictionary.
        If the table name also exists in the environment, ignore use_environment and call this function in priorty.
        """

        # False Positive: The function is called even when the argument is a table name which does not exist in the random table dictionary.
        if table_name not in self.random_tables:
            msg = f"UNEXPECTED FUNCTION CALLING: THE TABLE {table_name} DOES NOT EXIST IN THE RANDOM TABLE DICTIONARY."
            print_system_log(msg, after_break=True)
            return msg

        entries = self.random_tables[table_name]

        # If the table entries are empty.
        if len(entries) == 0:
            msg = f"THERE IS NOTHING IN {table_name}."
            print_system_log(msg, after_break=True)
            return msg
        if not self.automated_player:
            _ = input(f"THE RANDOM TABLE ACCESS: PRESS ANY KEY TO ROLL A DICE.")
        idx = random.randint(0, len(entries)-1)
        object_name = entries[idx]

        # The default system prompt consists of the instruction to generate the specific description of the object.
        system_prompt = ' '.join(GENERATE_OBJECT_DESC_PROMPT)
        kani = Kani(self.engine, chat_history=[self.scene_prompt], system_prompt=system_prompt)
        object_desc = await kani.chat_round_str(f"Generate the plausible description of the object.\nObject: {object_name}")

        # The default system prompt consists of the instruction to check if the object is obtainable.
        system_prompt = ' '.join(OBTAINABLE_CHECK_PROMPT)
        options = ['Obtainable', 'Not obtainable']
        options_str = '\n'.join([f"{o}: {option}" for o, option in enumerate(options)])
        kani = Kani(self.engine, chat_history=[self.scene_prompt], system_prompt=system_prompt)
        res = await kani.chat_round_str(f"Is this object obtainable which can be stored in the player inventory?\n{object_name}: {object_desc}\n\n{options_str}")
        res = convert_into_class_idx(res, options)

        self.function_arguments['player_name'] = player_name
        self.function_arguments['table_name'] = table_name
        self.function_intermediate_res[f"Generated description of the object '{object_name}'"] = object_desc
        self.function_intermediate_res[f"Obtainable object detection result for '{object_name}'"] = options[res]

        if res == 0:  # The item is obtainable.
            # Removing unnecessary punctuations from the object name.
            item_name = remove_punctuation(object_name)

            print_system_log(f"{item_name}: {object_desc}")
            print_system_log("ARE YOU GOING TO TAKE THIS ITEM?")
            selected = select_random_options(['Yes', 'No']) if self.automated_player else select_options(['Yes', 'No'])

            player_idx = self.name_to_idx[player_name]
            player = self.players[player_idx]

            if selected == 0:
                obtain_msg = self.add_item(player, item_name, object_desc)

                # Checking if the player took the item to update the random table.
                if item_name in player.inventory:
                    entries = entries[:idx] + entries[idx+1:]
                    self.random_tables[table_name] = entries

                msg = f"THE PLAYER {player_name} FOUND THE ITEM {item_name}.\n{obtain_msg}"
                print_system_log(msg, after_break=True)

                return msg
            
            msg = f"THE PLAYER {player_name} FOUND THE ITEM {item_name}, BUT DECIDED NOT TO TAKE IT."
            print_system_log(msg, after_break=True)

            return msg

        msg = f"THE PLAYER {player_name} FOUND {object_name}. IT SEEMS NOT OBTAINABLE."
        print_system_log(msg, after_break=True)

        return msg

    # Validating if the current interaction falls into the success condition.
    async def validate_success_condition(self):
        if len(self.success_condition) == 0:
            return False

        # The default system prompt consists of the instruction and the success condition.
        system_prompt = ' '.join(VALIDATE_SUCCESS_PROMPT)

        options = ['Succeeded', 'Not yet']
        options_str = '\n'.join([f"{o}: {option}" for o, option in enumerate(options)])
        kani = Kani(self.engine, chat_history=deepcopy(self.chat_history), system_prompt=system_prompt)
        res = await kani.chat_round_str(f"Have the players accomplished the success condition?\nSuccess condition: {self.success_condition}\n\n{options_str}")
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
        kani = Kani(self.engine, chat_history=deepcopy(self.chat_history), system_prompt=system_prompt)
        res = await kani.chat_round_str(f"Have the players fallen into the failure condition?\nFailure condition: {self.failure_condition}\n\n{options_str}")
        res = convert_into_class_idx(res, options)
        
        return True if res == 0 else False
