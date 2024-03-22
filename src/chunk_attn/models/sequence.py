from threading import Event
from typing import Any
import uuid


class Sequence:
    def __init__(self, prompt_tokens, max_tokens=2048) -> None:
        self.id = str(uuid.uuid4())
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.tokens = [x for x in self.prompt_tokens]
        self.index = None
        self.error = None
        self.result_ready = Event()

    def get_result(self) -> Any:
        self.result_ready.wait()
        if self.error is not None:
            raise self.error
        return self.tokens

    def set_result_ready(self):
        self.result_ready.set()

    def set_error(self, error: Exception):
        self.error = error
        self.result_ready.set()
    
    def append(self, token: int):
        self.tokens.append(token)
    
    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index]
