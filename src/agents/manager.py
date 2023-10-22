from kani import Kani, ai_function, AIParam
from kani.models import ChatMessage, ChatRole
from kani.exceptions import FunctionCallException, MessageTooLong
from agents.player import Player
from constants import VALIDATE_SUCCESS_PROMPT, VALIDATE_FAILURE_PROMPT, CREATE_NPC_PROMPT
from typing import Any, List, Dict, AsyncIterable, Annotated, Tuple
from copy import deepcopy

import json
import logging
import random

log = logging.getLogger("kani")
message_log = logging.getLogger("kani.messages")


# The whole game manager class.
class GameManager(Kani):
    def __init__(self, main_args, *args, **kwargs):
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
        self.environment = []
        self.random_tables = {}
        self.consequences = ""

        # Additional prompt attrbutes.
        self.player_prompts = []
        self.concat_policy = main_args.concat_policy
        self.max_turns = main_args.max_turns
        self.summarization = True if main_args.summarization else False
        self.summ_period = main_args.summ_period
        self.clear_raw_logs = True if main_args.clear_raw_logs else False

        # Additional attributes for game play.
        self.is_action_scene = False

    # Initialization of the scene.
    async def init_scene(self, init_query: str, scene: Dict[str, Any], **kwargs):
        query = f"{init_query}\n{scene}"
        res = await self.chat_round_str(query, **kwargs)

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

        # Including the initialized information into the fixed prompt.
        scene_prompt = f"[SCENE INFORMATION] [CHAPTER] {self.chapter} [SCENE] {self.scene} [SCENE_SUMMARY] {' '.join(self.scene_summary)} " + \
            f"[NPCS] {' '.join(self.get_npcs())} [GENERATION_RULES] {' '.join(self.get_generation_rules())} " + \
            f"[SUCCESS_CONDITION] {self.success_condition} [FAILURE_CONDITION] {self.failure_condition}" + \
            f"[GAME_FLOW] {' '.join(self.game_flow)} [ENVIRONMENT] {' '.join(self.environment)} " + \
            f"[RANDOM_TABLE] {' '.join(self.get_random_tables())} [CONSEQUENCES] {self.consequences}"
        self.always_included_messages.append(ChatMessage.system(scene_prompt))

    # Getter for NPCs with the natural format.
    def get_npcs(self):
        res = []
        for n, (name, info) in enumerate(self.npcs.items()):
            res.append(f"({n+1}) Name: {name} Kin: {info['kin']} Persona: {' '.join(info['persona'])} Goal: {info['goal']} Trait: {info['trait']} Flaw: {info['flaw']}")
        return res

    # Getter for generation rules with the natural format.
    def get_generation_rules(self):
        return [f"({r+1}) {rule}" for r, rule in enumerate(self.generation_rules)]

    # Getter for random tables with the natural format.
    def get_random_tables(self):
        res = []
        for table, entries in self.random_tables.items():
            res.append(f"{table}: ({', '.join(entries)[:-1]})")
        return res

    # Showing the scene information which the manager has initialized.
    def show_scene(self):
        print("<CHAPTER>")
        print(self.chapter)

        print("<SCENE>")
        print(self.scene)

        print("<SCENE SUMMARY>")
        print(self.scene_summary)

        print("<NPCS>")
        print(self.npcs)

        print("<GENERATION RULES>")
        print(self.generation_rules)

        print("<SUCCESS CONDITION>")
        print(self.success_condition)

        print("<FAILURE CONDITION>")
        print(self.failure_condition)

        print("<GAME FLOW>")
        print(self.game_flow)

        print("<ENVIRONMENT>")
        print(self.environment)

        print("<RANDOM TABLES>")
        print(self.random_tables)

        print("<CONSEQUENCES>")
        print(self.consequences)

    # Making a prompt using the simple concatenation.
    def get_simple_prompt(self, always_len: int) -> list[ChatMessage]:
        if self.max_turns is None:
            valid_chat_history = deepcopy(self.chat_history)
        else:
            valid_chat_history = self.chat_history[len(self.chat_history)-self.max_turns]

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
            return self.always_included_messages
        return self.always_included_messages + self.player_prompts + valid_chat_history[-to_keep:]

    # Overriding get_prompt.
    async def get_prompt(self) -> list[ChatMessage]:
        always_len = self.always_len
        for message in self.player_prompts:  # Additional length for player information.
            always_len += self.message_token_len(message)

        if self.summarization and self.summ_period is None:
            pass  # TODO: Summarizing all previous chat logs and return the prompt.

        if self.concat_policy == 'simple':
            return self.get_simple_prompt(always_len)

    # Overriding full_round.
    async def full_round(self, user_queries: List[Tuple[int, str]], players: Dict[int, Player], **kwargs) -> AsyncIterable[ChatMessage]:
        """Perform a full chat round (user -> model [-> function -> model -> ...] -> user).

        Yields each of the model's ChatMessages. A ChatMessage must have at least one of (content, function_call).

        Use this in an async for loop, like so::

            async for msg in kani.full_round("How's the weather?"):
                print(msg.content)

        :param query: The content of the user's chat message.
        :param kwargs: Additional arguments to pass to the model engine (e.g. hyperparameters).
        """
        # Converting the player information into natural language prompt.
        user_messages = []
        self.player_prompts.clear()
        for pair in user_queries:
            p, query = pair
            prompt = f"[Player {p}] [Name] {players[p].name} [Kin] {players[p].kin} [Persona] {' '.join(players[p].get_persona())} [Goal] {players[p].goal} " + \
                f"[Traits] {' '.join(players[p].get_traits())} [Flaws] {' '.join(players[p].get_flaws())} [Items] {' '.join(players[p].get_items())}"
            self.player_prompts.append(ChatMessage.system(prompt))
            user_messages.append(ChatMessage.user(content=query.strip(), name=players[p].name))

        retry = 0
        is_model_turn = True
        async with self.lock:
            # Adding the player name for multi-players setting.
            self.chat_history += user_messages

            while is_model_turn:
                # do the model prediction
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

    # Kani's function call for a dice roll test.
    @ai_function
    async def activate_test(self, difficulty: Annotated[int, AIParam(desc="The difficulty of the task in a range of 2 and 6.")]):
        """Activate the test if there is a test to be performed and let the player roll a dice."""
        _ = input(f"THE TEST DIFFICULTY: {difficulty}: PRESS ANY KEY TO ROLL A DICE.")
        res = random.randint(2, 6)

        if res < difficulty:
            msg = f"TEST FAILED. THE DICE ROLL RESULT IS: {res}."
            print(msg)
        else:
            msg = f"TEST SUCCEEDED. THE DICE ROLL RESULT IS: {res}."
            print(msg)
        
        # Updating the new chat message.
        msg = ChatMessage.system(content=msg)
        await self.add_to_history(msg)

    # Kani's function call for starting an action scene.
    @ai_function
    def activate_action_scene(self):
        """Activate an action scene if there is a circumstance that players should take actions in a tight time limit."""
        self.is_action_scene = True

    # Kani's function call for ending an action scene.
    @ai_function
    def terminate_action_scene(self):
        """Terminate the current ongoing action scene if an urgent circumstance has been finished."""
        self.is_action_scene = False

    # Kani's function call for creating an NPC immediately.
    @ai_function
    async def create_npc(self, name: Annotated[str, AIParam(desc="The name of the NPC which has been requested by the player.")]):
        """Create an NPC a player requested to talk with if it has not been initialized yet.""" 

        # The default system prompt consists of the instruction and the requirement for an NPC.
        system_prompt = ' '.join(CREATE_NPC_PROMPT)
        system_prompt += f"\nCurrently initialized NPCs: {self.npcs}"
        
        kani = Kani(self.engine, chat_history=deepcopy(self.chat_history), system_prompt=system_prompt)
        res = await kani.chat_round_str(f"Check if the requested NPC '{name}' has already been initialized. Generate the specifications of it if it does not exist now.")

        if not self.translate_into_binary(res):
            return

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
