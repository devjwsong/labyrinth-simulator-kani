from kani import Kani
from kani.models import ChatMessage, ChatRole
from kani.exceptions import FunctionCallException, MessageTooLong
from agents.player import Player
from constant_prompts import VALIDATE_SUCCESS_PROMPT, VALIDATE_FAILURE_PROMPT
from typing import Any, List, Dict, AsyncIterable

import json
import logging

log = logging.getLogger("kani")
message_log = logging.getLogger("kani.messages")


# The whole game manager class.
class GameManager(Kani):
    def __init__(self, *args, **kwargs):
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

        # Additional Kani attrbutes.
        self.player_prompts = []

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

    # Overriding get_prompt.
    async def get_prompt(self) -> list[ChatMessage]:
        """
        Called each time before asking the LM engine for a completion to generate the chat prompt.
        Returns a list of messages such that the total token count in the messages is less than
        ``(self.max_context_size - self.desired_response_tokens)``.

        Always includes the system prompt plus any always_included_messages at the start of the prompt.

        You may override this to get more fine-grained control over what is exposed in the model's memory at any given
        call.
        """
        always_len = self.always_len
        for message in self.player_prompts:  # Additional length for player information.
            always_len += self.message_token_len(message)

        remaining = max_size = self.max_context_size - always_len
        total_tokens = 0
        to_keep = 0  # messages to keep from the end of chat history
        for message in reversed(self.chat_history):
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
        return self.always_included_messages + self.player_prompts + self.chat_history[-to_keep:]

    # Overriding full_round.
    async def full_round(self, query: str, player: Player, **kwargs) -> AsyncIterable[ChatMessage]:
        """Perform a full chat round (user -> model [-> function -> model -> ...] -> user).

        Yields each of the model's ChatMessages. A ChatMessage must have at least one of (content, function_call).

        Use this in an async for loop, like so::

            async for msg in kani.full_round("How's the weather?"):
                print(msg.content)

        :param query: The content of the user's chat message.
        :param kwargs: Additional arguments to pass to the model engine (e.g. hyperparameters).
        """
        # Converting the player information into natural language prompt.
        self.player_prompts.clear()
        player_prompt = f"[Player 1] [Name] {player.name} [Kin] {player.kin} [Persona] {' '.join(player.get_persona())} [Goal] {player.goal} " + \
            f"[Traits] {' '.join(player.get_traits())} [Flaws] {' '.join(player.get_flaws())} [Items] {' '.join(player.get_items())}"
        self.player_prompts.append(ChatMessage.system(player_prompt))

        retry = 0
        is_model_turn = True
        async with self.lock:
            # Adding the player name for multi-players setting.
            await self.add_to_history(ChatMessage.user(query.strip(), name=player.name))

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

    # Converting the generation result into the binary answer.
    def translate_into_binary(self, response: str):
        if 'yes' in response.lower():
            return True
        elif 'no' in response.lower():
            return False
        else:
            return None

    # Validating the generated response from the NPC.
    async def validate_generation_rule(self, chat_history: List[ChatMessage]):
        # The default system prompt consists of the instruction and the predefined generation rules.
        system_prompt = "You are the game manager in a fantasy text adventure game. " + \
            "You should determine whether the last response from the NPC follows the defined rule. " + \
            "You are given the dialogue history so far between the player(user) and the NPC(assistant) for reference. " + \
            "You must answer only either 'yes' or 'no'. "
        system_prompt += "Rules: "
        for r, rule in enumerate(self.generation_rules):
            system_prompt += f"{r+1} - {rule} "
        system_prompt = system_prompt[:-1]

        kani = Kani(self.engine, chat_history=chat_history, system_prompt=system_prompt)
        response = await kani.chat_round_str("Does the last response follow the rules?")

        return self.translate_into_binary(response)

    # Validating if the current interaction falls into the success condition.
    async def validate_success_condition(self):
        if len(self.success_condition) == 0:
            return False

        # The default system prompt consists of the instruction and the success condition.
        system_prompt = ' '.join(VALIDATE_SUCCESS_PROMPT)
        system_prompt += f"\nSuccess condition: {self.success_condition}"

        kani = Kani(self.engine, chat_history=self.chat_history, system_prompt=system_prompt)
        response = await kani.chat_round_str("Have the player accomplished the success condition?")

        return self.translate_into_binary(response)
    
    # Validating if the current interaction falls into the failure condition.
    async def validate_failure_condition(self):
        if len(self.failure_condition) == 0:
            return False

        # The default system prompt consists of the instruction and the failure condition.
        system_prompt = ' '.join(VALIDATE_FAILURE_PROMPT)
        system_prompt += f"\nFailure condition: {self.failure_condition}"

        kani = Kani(self.engine, chat_history=self.chat_history, system_prompt=system_prompt)
        response = await kani.chat_round_str("Have the player fallen into the failure condition?")
        
        return self.translate_into_binary(response)
