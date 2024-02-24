from kani import Kani
from kani.models import ChatMessage, ChatRole
from kani.exceptions import MessageTooLong
from constants import RULE_SUMMARY
from copy import deepcopy

import logging
import warnings

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

        self.additional_notes = kwargs['additional_notes']

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

    # Getter for additional notes in a (numbered) list format.
    def get_additional_notes(self, with_number=False):
        if with_number:
            return [f"({s+1}) {sent}" for s, sent in enumerate(self.additional_notes)]
        return self.additional_notes

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

        print("ADDITIONAL NOTES")
        print('\n'.join(self.get_additional_notes(with_number=True)))
            
    # Adding a trait.
    def add_trait(self, trait, desc):
        self.traits[trait] = desc
    
    # Adding a flaw.
    def add_flaw(self, flaw, desc):
        self.flaws[flaw] = desc
    
    # Adding an item.
    def add_item(self, item, desc):
        self.inventory[item] = desc

    # Removing a trait.
    def remove_trait(self, trait):
        self.traits.pop(trait)

    # Removing a flaw.
    def remove_flaw(self, flaw):
        self.flaws.pop(flaw)

    # Removing an item.
    def remove_item(self, item):
        self.inventory.pop(item)


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
            f"additional_notes={self.additional_notes}"
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
            return default_prompt
        prompt = default_prompt + self.chat_history[-to_keep:]

        return prompt

    # Overrding chat_round.
    async def chat_round(self, queries: list[ChatMessage], **kwargs) -> ChatMessage:
        """Perform a single chat round (user -> model -> user, no functions allowed).

        This is slightly faster when you are chatting with a kani with no AI functions defined.

        :param queries: The list of the user's chat message.
        :param kwargs: Additional arguments to pass to the model engine (e.g. hyperparameters).
        :returns: The model's reply.
        """
        # warn if the user has functions defined and has not explicitly silenced them in this call
        if self.functions and "include_functions" not in kwargs:
            warnings.warn(
                f"You have defined functions in the body of {type(self).__name__} but chat_round() will not call"
                " functions. Use full_round() instead.\nIf this is intentional, use chat_round(...,"
                " include_functions=False) to silence this warning."
            )
        kwargs = {**kwargs, "include_functions": False}
        # do the chat round
        async with self.lock:
            self.make_player_prompt()

            # add the manager's responses into the chat history.
            for msg in queries:
                await self.add_to_history(msg)

            # and get a completion
            completion = await self.get_model_completion(**kwargs)
            message = completion.message
            message = ChatMessage.assistant(name=self.name, content=message.content)
            await self.add_to_history(message)

            return message

    # Overrding chat_round_str.
    async def chat_round_str(self, queries: list[ChatMessage], **kwargs) -> str:
        """Like :meth:`chat_round`, but only returns the text content of the message."""
        msg = await self.chat_round(queries, **kwargs)
        return msg.text
