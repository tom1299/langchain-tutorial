from dotenv import load_dotenv

import langchain.chat_models

from langchain_core.language_models import BaseChatModel
from typing import Optional

load_dotenv()


def init_chat_model(provider: str, tokens: Optional[int] = None, model_name: Optional[str] = None, **kwargs) -> BaseChatModel:
    model = None
    if provider == "OpenAI":
        if not model_name:
            model_name = "gpt-4.1"
        init_kwargs = {
            "timeout": 30,
            **kwargs,
        }
        # If tokens explicitly provided and caller did not override with max_tokens, use it
        if tokens is not None and "max_tokens" not in init_kwargs:
            init_kwargs["max_tokens"] = tokens

        model = langchain.chat_models.init_chat_model(model_name, **init_kwargs)

    elif provider == "Anthropic":
        if not model_name:
            model_name = "claude-sonnet-4-5-20250929"
        init_kwargs = {
            "timeout": 30,
            **kwargs,
        }
        if tokens is not None and "max_tokens" not in init_kwargs:
            init_kwargs["max_tokens"] = tokens

        model = langchain.chat_models.init_chat_model(model_name, **init_kwargs)
    return model
