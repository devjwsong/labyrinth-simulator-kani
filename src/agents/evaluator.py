from kani import Kani
from kani.models import ChatMessage, ChatRole
from kani.exceptions import MessageTooLong
from copy import deepcopy
from constants import RULE_SUMMARY

import logging

log = logging.getLogger("kani")
message_log = logging.getLogger("kani.messages")


# The evaluator class inherited from Kani.
class Evaluator(Kani):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        rule_content = '\n'.join([' '.join(part) for part in RULE_SUMMARY])
        self.rule_prompt = ChatMessage.system(name="Game_Rules", content=rule_content)
    
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
        rule_prompt_len = self.message_token_len(self.rule_prompt)
        remaining = max_size = self.max_context_size - always_len - rule_prompt_len
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

        default_prompt = self.always_included_messages + [self.rule_prompt]

        if not to_keep:
            return default_prompt
        return default_prompt + self.chat_history[-to_keep:]
