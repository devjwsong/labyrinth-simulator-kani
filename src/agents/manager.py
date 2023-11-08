from kani import Kani, ai_function, AIParam
from kani.models import ChatMessage, ChatRole, FunctionCall
from kani.exceptions import FunctionCallException, MessageTooLong, NoSuchFunction, WrappedCallException
from kani.internal import FunctionCallResult
from kani.utils.message_formatters import assistant_message_contents
from sentence_transformers import SentenceTransformer, util
from constants import SEP, VALIDATE_SUCCESS_PROMPT, VALIDATE_FAILURE_PROMPT, CREATE_NPC_PROMPT, OBTAINABLE_CHECK_PROMPT
from utils import print_system_log, select_options
from typing import Any, List, Dict, AsyncIterable, Annotated, Tuple, Callable
from argparse import Namespace
from copy import deepcopy

import json
import logging
import random
import numpy as np
import torch
import string

log = logging.getLogger("kani")
message_log = logging.getLogger("kani.messages")


# The whole game manager class.
class GameManager(Kani):
    def __init__(self, main_args: Namespace, encoder: SentenceTransformer,*args, **kwargs):
        super().__init__(*args, **kwargs)

        # Attributes which should be initialized before the game.
        self.chapter = ""
        self.scene = ""
        self.scene_summary = []
        self.npcs = {}
        self.generation_rules = []
        self.success_condition = ""
        self.failure_condition = ""
        self.game_flow = []
        self.environment = {}
        self.random_tables = {}
        self.consequences = ""

        # Additional prompt attrbutes.
        self.scene_prompt = None
        self.player_prompts = []
        self.encoder = encoder
        self.sent_embs = np.empty((0, encoder.get_sentence_embedding_dimension())) if self.encoder is not None else None
        self.concat_policy = main_args.concat_policy
        self.max_turns = main_args.max_turns
        self.summarization = True if main_args.summarization else False
        self.summ_period = main_args.summ_period
        self.clear_raw_logs = True if main_args.clear_raw_logs else False

        # Additional attributes for game play.
        self.players = {}
        self.name_to_idx = {}
        self.is_action_scene = False

    # Initialization of the scene.
    async def init_scene(self, init_query: str, scene: Dict[str, Any], **kwargs):
        query = f"{init_query}\n{scene}"
        res = await self.chat_round_str(query, include_functions=False, **kwargs)

        # Finding each key and mapping into the corresponding attribute.
        try:
            res = json.loads(res)

            self.chapter = scene['chapter']
            self.scene = scene['scene']
            self.scene_summary = res['scene_summary']
            self.npcs = res['npcs']
            self.generation_rules = res['generation_rules']
            self.success_condition = res['success_condition']
            self.failure_condition = res['failure_condition']
            self.game_flow = res['game_flow']
            self.environment = res['environment']
            self.random_tables = scene['random_tables']
            self.consequences = scene['consequences']

        except json.decoder.JSONDecodeError as e:
            log.debug(res)
            log.error(f"{e}: The output format cannot be converted into dict.")
            raise Exception()
            # TODO: Fixing from the model if there is a JSON parsing error.
        except KeyError as e:
            log.debug(res)
            log.error(f"{e}: Missing key.")
            raise Exception()

        # Initialization record should be removed.
        self.chat_history = []
        if self.sent_embs is not None:
            self.sent_embs = np.empty((0, self.encoder.get_sentence_embedding_dimension()))

    # Getter for NPC in a natural format.
    def get_npc(self, info):
        return f"Kin: {info['kin']} {SEP} Persona: {' '.join(info['persona'])} {SEP} Goal: {info['goal']} {SEP} Trait: {info['trait']} {SEP} Flaw: {info['flaw']}"

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

    # Overriding add_to_history.
    async def add_to_history(self, message: ChatMessage):
        self.chat_history.append(message)

        # Sentence embedding for the retrieval.
        if self.encoder is not None:
            # Converting the content into an informative form.
            content = f"[{message.role.value.upper()}]"
            if message.name:
                content += f" {message.name}:"
            if message.content:
                content += f" {message.content}"

            emb = self.encoder.encode(content)
            self.sent_embs = np.concatenate((self.sent_embs, np.expand_dims(emb, axis=0)))

            # The number of sentence embeddings and chat logs should always be identical.
            assert len(self.chat_history) == self.sent_embs.shape[0], "The sentence embeddings and chat histories are not synced."

    # Making a prompt using the simple concatenation.
    def get_simple_history(self) -> list[ChatMessage]:
        if self.max_turns is None:
            valid_chat_history = deepcopy(self.chat_history)
        else:
            valid_chat_history = self.chat_history[max(len(self.chat_history)-self.max_turns, 0):]

        return valid_chat_history

    # Making a prompt using the retrieval concatenation.
    def get_retrieval_history(self) -> list[ChatMessage]:
        # If this is the case, retrieval has no meaning.
        if len(self.chat_history) <= self.max_turns or self.max_turns <= len(self.player_prompts):
            return self.get_simple_history()
        
        # Calculating the max-pooled cosine similarities.
        top_n = self.max_turns - len(self.player_prompts)
        query_embs, cand_embs = self.sent_embs[-len(self.player_prompts):], self.sent_embs[:-len(self.player_prompts)]  # (P, d), (C, d)
        cos_sims = util.cos_sim(query_embs, cand_embs)  # (P, C)
        scores = torch.max(cos_sims, dim=0).values  # (C)

        # Sorting the candidate logs by the similarities.
        idxs = torch.sort(scores, descending=True).indices[:top_n]
        idxs = torch.sort(idxs).values
        valid_chat_history = [self.chat_history[:-len(self.player_prompts)][idx] for idx in idxs]
        valid_chat_history += self.chat_history[-len(self.player_prompts):]

        # Checking the length of the valid chat logs.
        assert len(valid_chat_history) == self.max_turns, "The number of sampled chat logs and the maximum number of turns are different."

        return valid_chat_history

    # Overriding get_prompt.
    async def get_prompt(self) -> list[ChatMessage]:
        scene_prompt_len = 0
        if self.scene_prompt is not None:
            scene_prompt_len = self.message_token_len(self.scene_prompt)
        always_len = self.always_len + scene_prompt_len  # Additional length for scene information.
        for message in self.player_prompts:  # Additional length for player information.
            always_len += self.message_token_len(message)

        if self.summarization and self.summ_period is None:
            pass  # TODO: Summarizing all previous chat logs and return the prompt.

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

        default_prompt = self.always_included_messages + [self.scene_prompt] + self.player_prompts if self.scene_prompt is not None else self.always_included_messages
        if not to_keep:
            return default_prompt
        return default_prompt + valid_chat_history[-to_keep:]

    # Making the scene prompt.
    def make_scene_prompt(self):
        npcs_prompt = ' '.join(self.get_npcs(with_number=True))
        generation_rules_prompt = ' '.join(self.get_generation_rules(with_number=True))
        game_flow_prompt = ' '.join(self.get_game_flow(with_number=True))
        environment_prompt = ' '.join(self.get_environment(with_number=True))
        random_tables_prompt = ' '.join(self.get_random_tables(with_number=True))
        prompt= f"[SCENE INFORMATION] (CHAPTER) {self.chapter} (SCENE) {self.scene} (SCENE_SUMMARY) {' '.join(self.scene_summary)} " + \
            f"(NPCS) {npcs_prompt} (GENERATION_RULES) {generation_rules_prompt} " + \
            f"(SUCCESS_CONDITION) {self.success_condition} (FAILURE_CONDITION) {self.failure_condition}" + \
            f"(GAME_FLOW) {game_flow_prompt} (ENVIRONMENT) {environment_prompt} " + \
            f"(RANDOM_TABLE) {random_tables_prompt} (CONSEQUENCES) {self.consequences}"
        self.scene_prompt = ChatMessage.system(prompt)

    # Making the player prompts.
    def make_player_prompts(self, participants):
        # Converting the player information into the natural language prompt.
        self.player_prompts.clear()
        for p in participants:
            persona_prompt = ' '.join(self.players[p].get_persona(with_number=True))
            traits_prompt = ' '.join(self.players[p].get_traits(with_number=True))
            flaws_prompt = ' '.join(self.players[p].get_flaws(with_number=True))
            inventory_prompt = ' '.join(self.players[p].get_inventory(with_number=True))
            prompt = f"[CURRENT STATE OF Player {p}] (NAME) {self.players[p].name} (KIN) {self.players[p].kin} (PERSONA) {persona_prompt} (GOAL) {self.players[p].goal} " + \
                f"(TRAITS) {traits_prompt} (FLAWS) {flaws_prompt} (INVENTORY) {inventory_prompt}"
            self.player_prompts.append(ChatMessage.system(prompt))

    # Overriding full_round.
    async def full_round(self, user_queries: List[Tuple[int, str]], **kwargs) -> AsyncIterable[ChatMessage]:
        """Perform a full chat round (user -> model [-> function -> model -> ...] -> user).

        Yields each of the model's ChatMessages. A ChatMessage must have at least one of (content, function_call).

        Use this in an async for loop, like so::

            async for msg in kani.full_round("How's the weather?"):
                print(msg.content)

        :param query: The content of the user's chat message.
        :param kwargs: Additional arguments to pass to the model engine (e.g. hyperparameters).
        """
        participants = []
        for pair in user_queries:
            p, query = pair
            participants.append(p)
            await self.add_to_history(ChatMessage.user(content=query.strip(), name=self.players[p].name))

        retry = 0
        is_model_turn = True
        async with self.lock:
            while is_model_turn:
                # do the model prediction
                self.make_scene_prompt()
                self.make_player_prompts(participants)
                
                completion = await self.get_model_completion(**kwargs)
                message = completion.message
                await self.add_to_history(message)
                yield message

                # if function call, do it and attempt retry if it's wrong
                if not message.function_call:
                    return

                try:
                    is_model_turn = await self.do_function_call(message.function_call)
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
        user_queries: List[Tuple[int, str]],
        message_formatter: Callable[[ChatMessage], str | None] = assistant_message_contents,
        **kwargs,
    ) -> AsyncIterable[str]:
        """Like :meth:`full_round`, but each yielded element is a str rather than a ChatMessage.

        :param query: The content of the user's chat message.
        :param message_formatter: A function that returns a string to yield for each message. By default, `
            `full_round_str`` yields the content of each assistant message.
        :param kwargs: Additional arguments to pass to the model engine (e.g. hyperparameters).
        """
        async for message in self.full_round(user_queries, **kwargs):
            if text := message_formatter(message):
                yield text

    # Kani's function call for a dice roll test.
    @ai_function
    async def activate_test(self, difficulty: Annotated[int, AIParam(desc="The difficulty of the task in a range of 2 and 6.")]):
        """Activate the test if there is a test to be performed and let the player roll a dice."""
        _ = input(f"THE TEST DIFFICULTY: {difficulty}: PRESS ANY KEY TO ROLL A DICE.")
        res = random.randint(2, 6)

        if res < difficulty:
            msg = f"TEST FAILED. THE DICE ROLL RESULT IS: {res}."
        else:
            msg = f"TEST SUCCEEDED. THE DICE ROLL RESULT IS: {res}."
        
        # Updating the new chat message.
        print_system_log(msg, after_break=True)
        return msg

    # Kani's function call for starting an action scene.
    @ai_function
    def activate_action_scene(self):
        """Activate an action scene if there is a circumstance that players should take actions in a tight time limit."""
        self.is_action_scene = True
        msg = "ACTION SCENE ACTIVATED."
        print_system_log(msg, after_break=True)
        return msg

    # Kani's function call for ending an action scene.
    @ai_function
    def terminate_action_scene(self):
        """Terminate the current ongoing action scene if an urgent circumstance has been finished."""
        self.is_action_scene = False
        msg = "ACTION SCENE TERMINATED."
        print_system_log(msg, after_break=True)
        return msg

    # Kani's function call for creating an NPC immediately.
    @ai_function
    async def create_npc(self, name: Annotated[str, AIParam(desc="The name of the NPC which has been requested by the player.")]):
        """Create an NPC a player requested to talk with if it has not been initialized yet.""" 

        # The NPC has been already initialized.
        if name in self.npcs:
            msg = "NPC ALREADY EXISTS. CONTINUING THE GAME..."
            print_system_log(msg, after_break=True)
            return msg

        # The default system prompt consists of the instruction and the requirement for an NPC.
        system_prompt = ' '.join(CREATE_NPC_PROMPT)
        system_prompt += f"\nCurrently initialized NPCs: {self.npcs}"
        
        kani = Kani(self.engine, chat_history=deepcopy(self.chat_history), system_prompt=system_prompt)
        res = await kani.chat_round_str(f"Generate the specifications of the requested NPC '{name}'.")

        # Converting & Fetching information.
        try:
            res = json.loads(res)

            assert isinstance(res['kin'], str), "THE KIN OF AN NPC IS NOT THE STRING TYPE."
            assert isinstance(res['persona'], list), "THE PERSONA OF AN NPC IS NOT THE LIST TYPE."
            assert isinstance(res['goal'], str), "THE GOAL OF AN NPC IS NOT THE STRING TYPE."
            assert isinstance(res['trait'], str), "THE TRAITS OF AN NPC IS NOT THE STRING TYPE."
            assert isinstance(res['flaw'], str), "THE FLAWS OF AN NPC IS NOT THE STRING TYPE."

            self.npcs[name] = res

        except json.decoder.JSONDecodeError as e:
            log.debug(res)
            log.error(f"{e}: The output format cannot be converted into dict.")
            raise Exception()
        except KeyError as e:
            log.debug(res)
            log.error(f"{e}: Missing key.")
            raise Exception()

        msg = f"NPC {name} CREATED: {self.get_npc(self.npcs[name])}"
        print_system_log(msg, after_break=True)
        return msg

    # Kani's function call for obtaining an item in the environment.
    @ai_function
    def obtain_item(self,
        player_name: Annotated[str, AIParam(desc="The name of the player charater who wants to get the item from the environment")],
        item_name: Annotated[str, AIParam(desc="The name of the item which the player wants to obtain.")]
    ):
        """Let the player obtain the item in the environment if the player requested to get the item in the environment."""
        print("#" * 10 + "DEBUG" + "#" * 10)
        print(player_name)
        print("#" * 10 + "DEBUG" + "#" * 10)
        print(item_name)

    # Kani's function call for removing an item in the player's inventory.
    @ai_function
    def remove_item(self,
        player_name: Annotated[str, AIParam(desc="The name of the player charater who wants to remove the item from the inventory.")],
        item_name: Annotated[str, AIParam(desc="The name of the item which the player wants to discard.")]
    ):
        """Let the player discard the item if the player requested to remove an item from the inventory."""
        player = self.players[self.name_to_idx[player_name]]

        # Checking if the item is in the inventory.
        if item_name not in player.inventory:
            msg = f"THE PLAYER {player_name} TRIED TO DISCARD THE ITEM {item_name} BUT THERE IS NO SUCH ITEM IN THE INVENTORY."
            print_system_log(msg, after_break=True)
            return msg

        # Removing the item from the inventory.
        desc = player.inventory[item_name]
        player.remove_item(item_name)

        # Discarded item is placed in the environment.
        self.environment[item_name] = desc

        msg = f"THE PLAYER {player_name} REMOVED THE ITEM {item_name} FROM THE INVENTORY."
        print_system_log("PLAYER INVENTORY UPDATED:")
        print('\n'.join(player.get_inventory(with_number=True)))
        print_system_log(msg, after_break=True)

        return msg

    # Kani's function call for getting access to an item in a random table.
    @ai_function
    async def use_random_table(self, 
        player_name: Annotated[str, AIParam(desc="The name of the player charater which tries to get an item if the random table is a list of obtainable items.")], 
        table_name: Annotated[str, AIParam(desc="The name of the table to be accessed.")]
    ):
        """
        Let the player use to a random table if the player tries to use something in any random table 
        or if a certain table should be referred to anytime during the game.
        """

        entries = self.random_tables[table_name]
        _ = input(f"THE RANDOM TABLE ACCESS: PRESS ANY KEY TO ROLL A DICE.")
        idx = random.randint(0, len(entries)-1)
        obj = entries[idx]

        # The default system prompt consists of the instruction to check if the object is obtainable.
        system_prompt = ' '.join(OBTAINABLE_CHECK_PROMPT)
        system_prompt += f"\n{self.scene_prompt.content}"
        
        kani = Kani(self.engine, chat_history=deepcopy(self.chat_history), system_prompt=system_prompt)
        response = await kani.chat_round_str(f"Can the player get the given object {obj} in the inventory?")

        msg = None
        if self.translate_into_binary(response):  # The item is obtainble.
            # Removing unnecessary punctuations from the object name.
            puncs = list(string.punctuation)
            cut_idx = len(obj)
            for i in range(len(obj)-1, -1, -1):
                if obj[i] in puncs:
                    cut_idx = i
                else:
                    break
            obj = obj[:cut_idx]

            item_desc = await kani.chat_round_str(f"Generate the plausible one sentence description of the item {obj}.")
            print_system_log(f"{obj}: {item_desc}")
            print_system_log("ARE YOU GOING TO TAKE THIS ITEM?")
            selected = select_options(['Yes', 'No'])

            player_idx = self.name_to_idx[player_name]
            player = self.players[player_idx]

            if selected == 0:
                if len(player.inventory) >= 6:  # The player inventory is already full.
                    print_system_log("YOUR INVENTORY IS ALREADY FULL. CHOOSE ONE ITEM TO DISCARD FROM THE INVENTORY OR DECIDE NOT TO TAKE THE CURRENT ITEM.")
                    options = [
                        "Discarding one item from the inventory.",
                        "Not taking the found item."
                    ]
                    selected = select_options(options)
                    if selected == 0:  # Discarding any item from the inventory.
                        print_system_log("WHICH ITEM ARE YOU GOING TO DISCARD?")
                        selected = select_options(player.get_inventory())
                        removal_target = list(player.inventory.keys())[selected]
                        self.remove_item(player_name, removal_target)

                        player.add_item(obj, item_desc)
                        entries = entries[:idx] + entries[idx+1:]
                        self.random_tables[table_name] = entries

                        msg = f"THE PLAYER {player_name} ADDED THE ITEM {obj} IN THE INVENTORY."
                        print_system_log("PLAYER INVENTORY UPDATED:")
                        print('\n'.join(player.get_inventory(with_number=True)))
                        print_system_log(msg, after_break=True)
                        return msg
                    else:  # Not taking the found item.
                        msg = f"THE PLAYER {player_name} FOUND THE ITEM {obj} BUT DECIDED NOT TO TAKE THE ITME {obj}."
                        print_system_log(msg, after_break=True)
                        return msg
                else:
                    # Updating the player inventory and removing the item from the random table.
                    player.add_item(obj, item_desc)
                    entries = entries[:idx] + entries[idx+1:]
                    self.random_tables[table_name] = entries

                    msg = f"THE PLAYER {player_name} FOUND THE ITEM {obj} AND ADDED IT IN THE INVENTORY."
                    print_system_log("PLAYER INVENTORY UPDATED:")
                    print('\n'.join(player.get_inventory(with_number=True)))
                    print_system_log(msg, after_break=True)
            else:
                msg = f"THE PLAYER {player_name} FOUND THE ITEM {obj} BUT DECIDED NOT TO TAKE THE ITME {obj}."
                print_system_log(msg, after_break=True)
        else:
            msg = f"THE PLAYER {player_name} FOUND {obj}. IT SEEMS NOT OBTAINABLE."
            print_system_log(msg, after_break=True)

        return msg

    # Converting the generation result into the binary answer.
    def translate_into_binary(self, response: str):
        if 'yes' in response.lower():
            return True
        elif 'no' in response.lower():
            return False
        else:
            return None

    # Validating if the current interaction falls into the success condition.
    async def validate_success_condition(self):
        if len(self.success_condition) == 0:
            return False

        # The default system prompt consists of the instruction and the success condition.
        system_prompt = ' '.join(VALIDATE_SUCCESS_PROMPT)
        system_prompt += f"\nSuccess condition: {self.success_condition}"

        kani = Kani(self.engine, chat_history=deepcopy(self.chat_history), system_prompt=system_prompt)
        response = await kani.chat_round_str("Have the player accomplished the success condition?")

        return self.translate_into_binary(response)
    
    # Validating if the current interaction falls into the failure condition.
    async def validate_failure_condition(self):
        if len(self.failure_condition) == 0:
            return False

        # The default system prompt consists of the instruction and the failure condition.
        system_prompt = ' '.join(VALIDATE_FAILURE_PROMPT)
        system_prompt += f"\nFailure condition: {self.failure_condition}"

        kani = Kani(self.engine, chat_history=deepcopy(self.chat_history), system_prompt=system_prompt)
        response = await kani.chat_round_str("Have the player fallen into the failure condition?")
        
        return self.translate_into_binary(response)
