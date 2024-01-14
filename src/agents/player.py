from kani import Kani
from kani.models import ChatMessage, ChatRole, ToolCall
from kani.exceptions import MessageTooLong, FunctionCallException
from kani.internal import ExceptionHandleResult
from kani.utils.message_formatters import assistant_message_contents
from argparse import Namespace
from constants import RULE_SUMMARY
from typing import AsyncIterable
from copy import deepcopy

import logging
import asyncio

from utils import print_system_log

log = logging.getLogger("kani")
message_log = logging.getLogger("kani.messages")


# Default Player class.
class Player():
    def __init__(self, **kwargs):
        # Player character info based on the sheet in the text book.
        self.name = kwargs['name']
        self.kin = kwargs['kin']
        self.persona = kwargs['persona']
        self.goal = kwargs['goal']

        self.traits = kwargs['traits']
        self.flaws = kwargs['flaws']
        self.inventory = kwargs['inventory']

        self.guide = kwargs['guide']

    # Getter for persona in a (numbered) list format.
    def get_persona(self, with_number=False):
        if with_number:
            return [f"({s+1}) {sent}" for s, sent in enumerate(self.persona)]
        return self.persona

    # Getter for traits in a (numbered) list format.
    def get_traits(self, with_number=False):
        if with_number:
            return [f"({t+1}) {trait} - {desc}" for t, (trait, desc) in enumerate(self.traits.items())]
        return [f"{trait} - {desc}" for trait, desc in self.traits.items()]

    # Getter for flaws in a (numbered) list format.
    def get_flaws(self, with_number=False):
        if with_number:
            return [f"({f+1}) {flaw} - {desc}" for f, (flaw, desc) in enumerate(self.flaws.items())]
        return [f"{flaw} - {desc}" for flaw, desc in self.flaws.items()]

    # Getter for inventory in a (numbered) list format.
    def get_inventory(self, with_number=False):
        if with_number:
            return [f"({i+1}) {item} - {desc}" for i, (item, desc) in enumerate(self.inventory.items())]
        return [f"{item} - {desc}" for item, desc in self.inventory.items()]

    # Printing the character sheet so far.
    def show_info(self):
        print(f"NAME: {self.name}")
        print(f"KIN: {self.kin}")
        
        print("PERSONA")
        print('\n'.join(self.get_persona(with_number=True)))

        print(f"GOAL: {self.goal}")

        print("TRAITS")
        print('\n'.join(self.get_traits(with_number=True)))

        print("FLAWS")
        print('\n'.join(self.get_flaws(with_number=True)))

        print("INVENTORY")
        print('\n'.join(self.get_inventory(with_number=True)))
            
    # Adding a trait.
    def add_trait(self, trait, desc):
        self.traits[trait] = desc
        
        # Updating the new chat message.
        msg = f"PLAYER {self.name.upper()} ADDED A TRAIT '{trait}: {desc}'"
        print_system_log(msg)
        return msg
    
    # Adding a flaw.
    def add_flaw(self, flaw, desc):
        self.flaws[flaw] = desc

        # Updating the new chat message.
        msg = f"PLAYER {self.name.upper()} ADDED A FLAW '{flaw}: {desc}'"
        print_system_log(msg)
        return msg
    
    # Adding an item.
    def add_item(self, item, desc):
        self.inventory[item] = desc

        # Updating the new chat message.
        msg = f"PLAYER {self.name.upper()} ADDED AN ITEM '{item}: {desc}' IN THE INVENTORY."
        print_system_log(msg)
        return msg

    # Removing a trait.
    def remove_trait(self, trait):
        if trait not in self.traits:
            print_system_log("NO SUCH TRAIT. BACK TO THE COMMMAND SELECTION.")
            return

        self.traits.pop(trait)

        # Updating the new chat message.
        msg = f"PLAYER {self.name.upper()} REMOVED THE TRAIT '{trait}'."
        print_system_log(msg)
        return msg

    # Removing a flaw.
    def remove_flaw(self, flaw):
        if flaw not in self.flaws:
            print_system_log("NO SUCH FLAW. BACK TO THE COMMMAND SELECTION.")
            return

        self.flaws.pop(flaw)

        # Updating the new chat message.
        msg = f"PLAYER {self.name.upper()} REMOVED THE FLAW '{flaw}'."
        print_system_log(msg)
        return msg

    # Removing an item.
    def remove_item(self, item):
        if item not in self.inventory:
            print_system_log("NO SUCH ITEM. BACK TO THE COMMMAND SELECTION.")
            return
        
        self.inventory.pop(item)

        # Updating the new chat message.
        msg = f"PLAYER {self.name.upper()} REMOVED THE ITEM '{item}'."
        print_system_log(msg)
        return msg


# Kani version of Player class.
class PlayerKani(Player, Kani):
    def __init__(self, *args, **kwargs):
        Player.__init__(self, **kwargs)
        Kani.__init__(self, engine=kwargs['engine'], system_prompt=kwargs['system_prompt'])

        # Context prompts.
        rule_content = '\n'.join([' '.join(part) for part in RULE_SUMMARY])
        self.rule_prompt = ChatMessage.system(name="Game_Rules", content=rule_content)
        self.player_prompt = None

    # Making the player prompt.
    def make_player_prompt(self):
        content = f"name={self.name}, kin={self.kin}, persona={self.persona}, goal={self.goal}, " + \
            f"traits={self.traits}, flaws={self.flaws}, inventory={self.inventory}, " + \
            f"kin_specific_featur={self.guide}"
        self.player_prompt = ChatMessage.system(name="Player_State", content=content)

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
        rule_prompt_len = self.message_token_len(self.rule_prompt)
        player_prompt_len = self.message_token_len(self.player_prompt)
        always_len = self.always_len + rule_prompt_len + player_prompt_len

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
                    f"{func_help}Content: {message.text[:100]}..."
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
        if self.player_prompt is not None:
            default_prompt += [self.player_prompt]

        if not to_keep:
            return self.always_included_messages
        return self.always_included_messages + self.chat_history[-to_keep:]


    # Overriding full_round.
    async def full_round(self, ai_queries: list[ChatMessage], **kwargs) -> AsyncIterable[ChatMessage]:
        """Perform a full chat round (user -> model [-> function -> model -> ...] -> user).

        Yields each of the model's ChatMessages. A ChatMessage must have at least one of (content, function_call).

        Use this in an async for loop, like so::

            async for msg in kani.full_round("How's the weather?"):
                print(msg.content)

        :param query: The content of the user's chat message.
        :param kwargs: Additional arguments to pass to the model engine (e.g. hyperparameters).
        """
        for msg in ai_queries:
            await self.add_to_history(msg)

        retry = 0
        is_model_turn = True
        async with self.lock:
            while is_model_turn:
                # Setting the player prompt.
                self.make_player_prompt()

                # do the model prediction
                completion = await self.get_model_completion(**kwargs)
                message = completion.message
                await self.add_to_history(message)
                yield message

                # if function call, do it and attempt retry if it's wrong
                if not message.tool_calls:
                    return

                # run each tool call in parallel
                async def _do_tool_call(tc: ToolCall):
                    try:
                        return await self.do_function_call(tc.function, tool_call_id=tc.id)
                    except FunctionCallException as e:
                        return await self.handle_function_call_exception(tc.function, e, retry, tool_call_id=tc.id)

                # and update results after they are completed
                is_model_turn = False
                should_retry_call = False
                n_errs = 0
                results = await asyncio.gather(*(_do_tool_call(tc) for tc in message.tool_calls))
                for result in results:
                    # save the result to the chat history
                    await self.add_to_history(result.message)
                    yield result.message
                    if isinstance(result, ExceptionHandleResult):
                        is_model_turn = True
                        n_errs += 1
                        # retry if any function says so
                        should_retry_call = should_retry_call or result.should_retry
                    else:
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
