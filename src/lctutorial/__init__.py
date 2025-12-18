from dotenv import load_dotenv

import langchain.chat_models

from langchain_core.language_models import BaseChatModel

load_dotenv()

def init_chat_model(provider: str, tokens: int) -> BaseChatModel:
    model = None
    if provider == "OpenAI":
        model = langchain.chat_models.init_chat_model("gpt-4.1",
                                       timeout=30,
                                       max_tokens=tokens
                                       # use_responses_api=True
                                       )
    elif provider == "Anthropic":
        model = langchain.chat_models.init_chat_model("claude-sonnet-4-5-20250929",
                                          temperature=0.7,
                                          timeout=30,
                                          max_tokens=tokens
                                          )
    return model