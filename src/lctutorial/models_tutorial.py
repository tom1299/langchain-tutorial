""""
From https://docs.langchain.com/oss/python/langchain/models#openai

TODO: Refactor this code into multiple files, use mock llms where applicable.
"""
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain.chat_models import init_chat_model


# AI generated code
def print_token_usage(model_response: AIMessage) -> None:
    usage = model_response.usage_metadata
    if usage:
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        print(f"Input Tokens: {input_tokens}, Output Tokens: {output_tokens}, Total Tokens: {total_tokens}")
    else:
        print("No usage metadata available.")


load_dotenv()

openai_model = init_chat_model("gpt-4.1",
                               temperature=0.7,
                               timeout=30,
                               max_tokens=1000,  # Anthropic min tokens
                               use_responses_api=True  # OpenAI specific parameter
                               )

response: AIMessage = openai_model.invoke("Why do parrots talk?")

print(response.content)
print_token_usage(response)

anthropic_model = init_chat_model("claude-sonnet-4-5-20250929",
                                  temperature=0.7,
                                  timeout=30,
                                  max_tokens=10000,
                                  )

response = openai_model.invoke("Why do parrots talk?")

content = response.content
if isinstance(content, list):
    # Check length of list and if first element is a dict with 'text' key
    if len(content) > 0 and isinstance(content[0], dict) and "text" in content[0]:
        print(content[0]["text"])
    else:
        print("Unexpected content format:", content)
else:
    print(content)                 # Content is a str

print_token_usage(response)

conversation = [
    {"role": "system", "content": "You are a helpful assistant that translates English to French."},
    {"role": "user", "content": "Translate: I love programming."},
    {"role": "assistant", "content": "J'adore la programmation."},
    {"role": "user", "content": "Translate: I love building applications."}
]

response = anthropic_model.invoke(conversation)
print(response.content)         # Content is of type str

full: AIMessageChunk = None  # None | AIMessageChunk
for chunk in anthropic_model.stream("What color is the sky?"):
    chunk: AIMessageChunk
    full = chunk if full is None else full + chunk
    for block in chunk.content_blocks:
        if block["type"] == "reasoning" and (reasoning := block.get("reasoning")):
            print(f"Reasoning: {reasoning}")
        elif block["type"] == "tool_call_chunk":
            print(f"Tool call chunk: {block}")
        elif block["type"] == "text":
            print(block["text"])

print(full.content_blocks)

for response in anthropic_model.batch_as_completed([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
], config={
    'max_concurrency': 2,  # Limit to 2 parallel calls
}):
    print(response)

# Tools
# TODO: Look at built-in tools in Anthropic and OpenAI integrations

from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."

openai_model_with_tools = openai_model.bind_tools([get_weather])

openai_model_called_tools = False
response = openai_model_with_tools.invoke("What's the weather like in Boston?")
print(response)
for tool_call in response.tool_calls:
    # View tool calls made by the model
    openai_model_called_tools = True
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")

print(openai_model_called_tools)    # True

anthropic_model_called_tools = False
anthropic_model_with_tools = anthropic_model.bind_tools([get_weather], tool_choice="get_weather") # Specify tool_choice to force tool usage
response = anthropic_model_with_tools.invoke("What's the weather like in Boston?")
# !!! Response only contains tool calls if the model decides to use them
for tool_call in response.tool_calls:
    # View tool calls made by the model
    anthropic_model_called_tools = True
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")

print(anthropic_model_called_tools)     # False TODO: Why does the anthropic model not call the tool? => max_tokens too low!


# Example with manual tool execution and getting final answer
# Bind (potentially multiple) tools to the model

# Step 1: Model generates tool calls
messages = [{"role": "user", "content": "What's the weather in Boston?"}]
ai_msg = anthropic_model_with_tools.invoke(messages)
messages.append(ai_msg)                                 # Append AI message with tool calls for later reference

# Step 2: Execute tools and collect results
for tool_call in ai_msg.tool_calls:
    # Execute the tool with the generated arguments
    tool_result = get_weather.invoke(tool_call)         # Correlated by tool id
    messages.append(tool_result)

# Step 3: Pass results back to model for final response
# Note: The model used is a different instance without tools bound
final_response = anthropic_model.invoke(messages)
print(final_response.text)          # TODO: Update docs with correct final response "The weather in Boston is currently sunny!"

# Continue here: https://docs.langchain.com/oss/python/langchain/models#streaming-tool-calls
# with parallel tool calls example

