# LLM

## Introduction

The LLM (Large Language Model) module provides a unified interface for interacting with various language model providers in the EvoAgentX framework. It abstracts away provider-specific implementation details, offering a consistent API for generating text, managing costs, and handling responses.

## Supported LLM Providers

EvoAgentX currently supports the following LLM providers:

### OpenAILLM

The primary implementation for accessing OpenAI's language models. It handles authentication, request formatting, and response parsing for models like GPT-4, GPT-3.5-Turbo, and other OpenAI models.

**Basic Usage:**

```python
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from utils.config import client_rotator

# Configure the model
client_config = client_rotator.get_next_client_config()
config = OpenAILLMConfig(
    model=client_config.model,
    openai_key=client_config.api_key,
    base_url=client_config.base_url,
    proxy=client_config.proxy,
    temperature=0.7,
    max_tokens=1000
)

# Initialize the model
llm = OpenAILLM(config=config)

# Generate text
response = llm.generate(
    prompt="Explain quantum computing in simple terms.",
    system_message="You are a helpful assistant that explains complex topics simply."
)
```

### Local LLM

We now support calling local models for your tasks, built on the LiteLLM framework for a familiar user experience. For example, to use Ollama, follow these steps:

1. Download the desired model, such as `ollama3`.
2. Run the model locally.
3. Configure the settings by specifying `api_base` (typically port `11434`) and setting `is_local` to `True`.

You're now ready to leverage your local model seamlessly!

**Basic Usage:**

```python

from evoagentx.models.model_configs import LiteLLMConfig
from evoagentx.models import LiteLLM

# use local model
config = LiteLLMConfig(
    model="ollama/llama3",
    api_base="http://localhost:11434",
    is_local=True,
    temperature=0.7,
    max_tokens=1000,
    output_response=True
)

# Generate 
llm = LiteLLM(config)
response = llm.generate(prompt="What is Agentic Workflow?")

```


## Core Functions

All LLM implementations in EvoAgentX provide a consistent set of core functions for generating text and managing the generation process.

### Generate Function

The `generate` function is the primary method for producing text with language models:

```python
def generate(
    self,
    prompt: Optional[Union[str, List[str]]] = None,
    system_message: Optional[Union[str, List[str]]] = None,
    messages: Optional[Union[List[dict],List[List[dict]]]] = None,
    parser: Optional[Type[LLMOutputParser]] = None,
    parse_mode: Optional[str] = "json", 
    parse_func: Optional[Callable] = None,
    **kwargs
) -> Union[LLMOutputParser, List[LLMOutputParser]]:
    """
    Generate text based on the prompt and optional system message.

    Args:
        prompt: Input prompt(s) to the LLM.
        system_message: System message(s) for the LLM.
        messages: Chat message(s) for the LLM, already in the required format (either `prompt` or `messages` must be provided).
        parser: Parser class to use for processing the output into a structured format.
        parse_mode: The mode to use for parsing, must be the `parse_mode` supported by the `parser`. 
        parse_func: A function to apply to the parsed output.
        **kwargs: Additional generation configuration parameters.
        
    Returns:
        For single generation: An LLMOutputParser instance.
        For batch generation: A list of LLMOutputParser instances.
    """
```

#### Inputs 

In EvoAgentX, there are several ways to provide inputs to LLMs using the `generate` function:

**Method 1: Prompt and System Message**

1. **Prompt**: The specific query or instruction for which you want a response. 

2. **System Message** (optional): Instructions that guide the model's overall behavior and role. This sets the context for how the model should respond.

Together, these components are converted into a standardized message format that the language model can understand:

```python
# Simple example with prompt and system message
response = llm.generate(
    prompt="What are three ways to improve productivity?",
    system_message="You are a productivity expert providing concise, actionable advice."
)
```

Behind the scenes, this gets converted into messages with appropriate roles:

```python
messages = [
    {"role": "system", "content": "You are a productivity expert providing concise, actionable advice."},
    {"role": "user", "content": "What are three ways to improve productivity?"}
]
```

**Method 2: Using Messages Directly**

For more complex conversations or when you need precise control over the message format, you can use the `messages` parameter directly:

```python
# Using messages directly for a multi-turn conversation
response = llm.generate(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"},
        {"role": "assistant", "content": "I'm an AI assistant designed to help with various tasks."},
        {"role": "user", "content": "Can you help me with programming?"}
    ]
)
```

#### Batch Generation 

For batch processing, you can provide lists of prompts/system messages or list of messages. For example: 

```python
# Batch processing example
responses = llm.generate(
    prompt=["What is machine learning?", "Explain neural networks."],
    system_message=["You are a data scientist.", "You are an AI researcher."]
)
```

##### Parse Modes

EvoAgentX supports several parsing strategies:

1. **"str"**: Uses the raw output as-is for each field defined in the parser.
2. **"json"** (default): Extracts fields from a JSON string in the output.
3. **"xml"**: Extracts content from XML tags matching field names.
4. **"title"**: Extracts content from markdown sections (default format: "## {title}").
5. **"custom"**: Uses a custom parsing function specified by `parse_func`.

!!! note 
    For `'json'`, `'xml'` and `'title'`, you should instruct the LLM (through the `prompt`) to output the content in the specified format that can be parsed by the parser. Otherwise, the parsing will fail. 

    1. For `'json'`, you should instruct the LLM to output a valid JSON string containing keys that match the field names in the parser class. If there are multiple JSON string in the raw LLM output, only the first one will be parsed.  

    2. For `xml`, you should instruct the LLM to output content that contains XML tags matching the field names in the parser class, e.g., `<{field_name}>...</{field_name}>`. If there are multiple XML tags with the same field name, only the first one will be used. 

    3. For `title`, you should instruct the LLM to output content that contains markdown sections with the title exactly matching the field names in the parser class. The default title format is "## {title}". You can change it by setting the `title_format` parameter in the `generate` function, e.g., `generate(..., title_format="### {title}")`. The `title_format` must contain `{title}` as a placeholder for the field name.  

##### Custom Parsing Function

For maximum flexibility, you can define a custom parsing function with `parse_func`:

```python
from evoagentx.models import LLMOutputParser
from evoagentx.core.module_utils import extract_code_block

class CodeOutput(LLMOutputParser):
    code: str = Field(description="The generated code")

# Use custom parsing
response = llm.generate(
    prompt="Write a Python function to calculate Fibonacci numbers.",
    parser=CodeOutput,
    parse_mode="custom",
    parse_func=lambda content: {"code": extract_code_block(content)[0]}
)
```

!!! note 
    The `parse_func` should have an input parameter `content` that receives the raw LLM output, and return a dictionary with keys matching the field names in the parser class.  

### Async Generate Function

For applications requiring asynchronous operation, the `async_generate` function provides the same functionality as the `generate` function, but in a non-blocking manner:

```python
async def async_generate(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        system_message: Optional[Union[str, List[str]]] = None,
        messages: Optional[Union[List[dict],List[List[dict]]]] = None,
        parser: Optional[Type[LLMOutputParser]] = None,
        parse_mode: Optional[str] = "json", 
        parse_func: Optional[Callable] = None,
        **kwargs
    ) -> Union[LLMOutputParser, List[LLMOutputParser]]:
    """
    Asynchronously generate text based on the prompt and optional system message.

    Args:
        prompt: Input prompt(s) to the LLM.
        system_message: System message(s) for the LLM.
        messages: Chat message(s) for the LLM, already in the required format (either `prompt` or `messages` must be provided).
        parser: Parser class to use for processing the output into a structured format.
        parse_mode: The mode to use for parsing, must be the `parse_mode` supported by the `parser`. 
        parse_func: A function to apply to the parsed output.
        **kwargs: Additional generation configuration parameters.
        
    Returns:
        For single generation: An LLMOutputParser instance.
        For batch generation: A list of LLMOutputParser instances.
    """
```

### Streaming Responses

EvoAgentX supports streaming responses from LLMs, which allows you to see the model's output as it's being generated token by token, rather than waiting for the complete response. This is especially useful for long-form content generation or providing a more interactive experience.

There are two ways to enable streaming:

#### Configure Streaming in the LLM Config

You can enable streaming when initializing the LLM by setting appropriate parameters in the config:

```python
# Enable streaming at initialization time
config = OpenAILLMConfig(
    model="gpt-4o-mini",
    openai_key="your-api-key",
    stream=True,  # Enable streaming
    output_response=True  # Print tokens to console in real-time
)

llm = OpenAILLM(config=config)

# All calls to generate() will now stream by default
response = llm.generate(
    prompt="Write a story about space exploration."
)
```

#### Enable Streaming in the Generate Method

Alternatively, you can enable streaming for specific generate calls:

```python
# LLM initialized with default non-streaming behavior
config = OpenAILLMConfig(
    model="gpt-4o-mini",
    openai_key="your-api-key"
)

llm = OpenAILLM(config=config)

# Override for this specific call
response = llm.generate(
    prompt="Write a story about space exploration.",
    stream=True,  # Enable streaming for this call only
    output_response=True  # Print tokens to console in real-time
)
```

# Agent

## Introduction

The `Agent` class is the fundamental building block for creating intelligent AI agents within the EvoAgentX framework. It provides a structured way to combine language models with actions, and memory management. 

## Architecture

An Agent consists of several key components:

1. **Large Language Model (LLM)**: 

    The LLM is specified through the `llm` or `llm_config` parameter and serve as the building block for the agent. It is responsible for interpreting context, generating responses, and making high-level decisions. The LLM will be passed to an action for executing a specific task. 

2. **Actions**: 

    Actions are the fundamental operational units of an agent. Each Action encapsulates a specific task and is the actual point where the LLM is invoked to reason, generate, or make decisions. While the Agent provides overall orchestration, it is through Actions that the LLM performs its core functions. Each Action is designed to do exactly one thing—such as retrieving knowledge, summarizing input, or calling an API—and can include the following components:

    - **prompt**: The prompt template used to guide the LLM's behavior for this specific task.
    - **inputs_format**: The expected structure and keys of the inputs passed into the action.
    - **outputs_format**: The format used to interpret and parse the LLM's output.
    - **tools**: Optional tools that can be integrated and utilized within the action.

3. **Memory Components**:

    Memory allows the agent to retain and recall relevant information across interactions, enhancing contextual awareness. There are two types of memory within the EvoAgentX framework: 

    - **Short-term memory**: Maintains the intermediate conversation or context for the current task. 
    - **Long-term memory (optional)**: Stores persistent knowledge that can span across sessions or tasks. This enables the agent to learn from past experiences, maintain user preferences, or build knowledge bases over time.


## Usage

### Basic Agent Creation

In order to create an agent, you need to define the actions that the agent will perform. Each action is defined as a class that inherits from the `Action` class. The action class should define the following components: `name`, `description`, `prompt`, `inputs_format`, and `outputs_format`, and implement the `execute` method (and `async_exectue` if you want to use the agent asynchronously). 


```python
from evoagentx.agents import Agent
from evoagentx.models import OpenAILLMConfig
from evoagentx.actions import Action, ActionInput, ActionOutput

# Define a simple action that uses the LLM to answer a question

class AnswerQuestionInput(ActionInput):
    question: str

class AnswerQuestionOutput(ActionOutput):
    answer: str

class AnswerQuestionAction(Action):

    def __init__(
        self, 
        name = "answer_question",
        description = "Answers a factual question using the LLM",   
        prompt = "Answer the following question as accurately as possible:\n\n{question}",
        inputs_format = AnswerQuestionInput,
        outputs_format = AnswerQuestionOutput,
        **kwargs
    ):
        super().__init__(
            name=name, 
            description=description, 
            prompt=prompt, 
            inputs_format=inputs_format, 
            outputs_format=outputs_format, 
            **kwargs
        )
    
    def execute(self, llm, inputs, sys_msg = None, return_prompt = False, **kwargs) -> AnswerQuestionOutput:
        question = inputs.get("question")
        prompt = self.prompt.format(question=question)
        response = llm.generate(
            prompt=prompt, 
            system_message=sys_msg,
            parser=self.outputs_format, 
            parse_mode="str"
        )

        if return_prompt:
            return response, prompt
        return response 

    async def async_execute(self, llm, inputs, sys_msg = None, return_prompt = False, **kwargs) -> AnswerQuestionOutput:
        question = inputs.get("question")
        prompt = self.prompt.format(question=question)
        response = await llm.async_generate(
            prompt=prompt, 
            system_message=sys_msg,
            parser=self.outputs_format, 
            parse_mode="str"
        )   
        if return_prompt:
            return response, prompt
        return response 

# Configure LLM
llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="your-api-key")

# Create an agent
agent = Agent(
    name="AssistantAgent",
    description="Answers a factual question using the LLM",
    llm_config=llm_config,
    system_prompt="You are a helpful assistant.",
    actions=[AnswerQuestionAction()]
)
```

### Executing Actions

You can directly call the `Agent` instance like a function. This will internally invoke the `execute()` method of the matching action using the specified `action_name` and `action_input_data`.

```python
# Execute an action with input data
message = agent(
    action_name="answer_question",
    action_input_data={"question": "What is the capital of France?"}
)

# Access the output
result = message.content.answer 
```

### Asynchronous Execution

You can also call the `Agent` instance in an asynchronous context. If the action defines an `async_execute` method, it will be used automatically when you `await` the agent.

```python
# Execute an action asynchronously
import asyncio 

async def main():
    message = await agent(
        action_name="answer_question",
        action_input_data={"question": "What is the capital of France?"}
    )
    return message.content.answer 

result = asyncio.run(main())
print(result)
```

## Memory Management

The Agent maintains a short-term memory for tracking conversation context:

```python
# Access the agent's memory
messages = agent.short_term_memory.get(n=5)  # Get last 5 messages

# Clear memory
agent.clear_short_term_memory()
```

## Agent Profile

You can get a human-readable description of an agent and its capabilities:

```python
# Get description of all actions
profile = agent.get_agent_profile()
print(profile)

# Get description of specific actions
profile = agent.get_agent_profile(action_names=["answer_question"])
print(profile)
```

## Prompt Management

Access and modify the prompts used by an agent:

```python
# Get all prompts
prompts = agent.get_prompts()
# prompts is a dictionary with the structure:
# {'answer_question': {'system_prompt': 'You are a helpful assistant.', 'prompt': 'Answer the following question as accurately as possible:\n\n{question}'}}

# Set a specific prompt
agent.set_prompt(
    action_name="answer_question",
    prompt="Please provide a clear and concise answer to the following query:\n\n{question}",
    system_prompt="You are a helpful assistant." # optional, if not provided, the system prompt will remain unchanged 
)

# Update all prompts
prompts_dict = {
    "answer_question": {
        "system_prompt": "You are an expert in providing concise, accurate information.",
        "prompt": "Please answer this question with precision and clarity:\n\n{question}"
    }
}
agent.set_prompts(prompts_dict)
```

## Saving and Loading Agents

Agents can be persisted and reloaded:

```python
# Save agent
agent.save_module("./agents/my_agent.json")

# Load agent (requires providing llm_config again)
loaded_agent = Agent.from_file(
    "./agents/my_agent.json", 
    llm_config=llm_config
)
```

## Context Extraction

The Agent includes a built-in context extraction mechanism that automatically derives appropriate inputs for actions from conversation history:

```python
# Context is automatically extracted when executing without explicit input data
response = agent.execute(
    action_name="action_name",
    msgs=conversation_history
)

# Get action inputs manually
action = agent.get_action("action_name")
inputs = agent.get_action_inputs(action)
```
# CustomizeAgent

## Introduction

The `CustomizeAgent` class provides a flexible framework for creating specialized LLM-powered agents. It enables the definition of agents with well-defined inputs, outputs, custom prompt templates, and configurable parsing strategies, making it suitable for rapid prototyping and deployment of domain-specific agents.

## Key Features

- **No Custom Code Required**: Create specialized agents through configuration rather than writing custom agent classes
- **Flexible Input/Output Definitions**: Define exactly what inputs your agent accepts and what outputs it produces
- **Customizable Parsing Strategies**: Multiple parsing modes to extract structured data from LLM responses
- **Reusable Components**: Save and load agent definitions for reuse across projects

## Basic Usage


### Simple Agent

The simplest way to create a `CustomizeAgent` is with just a name, description and prompt:

```python
import os 
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig
from evoagentx.agents import CustomizeAgent

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure LLM
openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)

# Create a simple agent
simple_agent = CustomizeAgent(
    name="SimpleAgent",
    description="A basic agent that responds to queries",
    prompt="Answer the following question: {question}",
    llm_config=openai_config,
    inputs=[
        {"name": "question", "type": "string", "description": "The question to answer"}
    ]
)

# Execute the agent
response = simple_agent(inputs={"question": "What is a language model?"})
print(response.content.content)  # Access the raw response content
```
In this example, 1. We specify the input information (including its name, type, and description) in the `inputs` parameter since the prompt requires an input. 2. Moreover, when executing the agent with `simple_agent(...)`, you should provide all the inputs in the `inputs` parameter. 

The output after executing the agent is a `Message` object, which contains the raw LLM response in `message.content.content`. 

!!! note
    All the input names specified in the `CustomizeAgent(inputs=[...])` should appear in the `prompt`. Otherwise, an error will be raised.


### Structured Outputs 

One of the most powerful features of `CustomizeAgent` is the ability to define structured outputs. This allows you to transform unstructured LLM responses into well-defined data structures that are easier to work with programmatically.

#### Basic Structured Output

Here's a simple example of defining structured outputs:

```python
from evoagentx.core.module_utils import extract_code_blocks

# Create a CodeWriter agent with structured output
code_writer = CustomizeAgent(
    name="CodeWriter",
    description="Writes Python code based on requirements",
    prompt="Write Python code that implements the following requirement: {requirement}",
    llm_config=openai_config,
    inputs=[
        {"name": "requirement", "type": "string", "description": "The coding requirement"}
    ],
    outputs=[
        {"name": "code", "type": "string", "description": "The generated Python code"}
    ],
    parse_mode="custom",  # Use custom parsing function
    parse_func=lambda content: {"code": extract_code_blocks(content)[0]}  # Extract first code block
)

# Execute the agent
message = code_writer(
    inputs={"requirement": "Write a function that returns the sum of two numbers"}
)
print(message.content.code)  # Access the parsed code directly
```

In this example:
1. We define an output field named `code` in the `outputs` parameter.
2. We set `parse_mode="custom"` to use a custom parsing function.
3. The `parse_func` extracts the first code block from the LLM response.
4. We can directly access the parsed code with `message.content.code`.

You can also access the raw LLM response by `message.content.content`. 

!!! note 
    1. If the `outputs` parameter is set in `CustomizeAgent`, the agent will try to parse the LLM response based on the output field names. If you don't want to parse the LLM response, you should not set the `outputs` parameter. The raw LLM response can be accessed by `message.content.content`. 

    2. CustomizeAgent supports different parsing modes, such as `['str', 'json', 'xml', 'title', 'custom']. Please refer to the [Parsing Modes](#parsing-modes) section for more details. 

#### Multiple Structured Outputs

You can define multiple output fields to create more complex structured data:

```python
# Agent that generates both code and explanation
analyzer = CustomizeAgent(
    name="CodeAnalyzer",
    description="Generates and explains Python code",
    prompt="""
    Write Python code for: {requirement}
    
    Provide your response in the following format:
    
    ## code
    [Your code implementation here]
    
    ## explanation
    [A brief explanation of how the code works]
    
    ## complexity
    [Time and space complexity analysis]
    """,
    llm_config=openai_config,
    inputs=[
        {"name": "requirement", "type": "string", "description": "The coding requirement"}
    ],
    outputs=[
        {"name": "code", "type": "string", "description": "The generated Python code"},
        {"name": "explanation", "type": "string", "description": "Explanation of the code"},
        {"name": "complexity", "type": "string", "description": "Complexity analysis"}
    ],
    parse_mode="title"  # Use default title parsing mode
)

# Execute the agent
result = analyzer(inputs={"requirement": "Write a binary search algorithm"})

# Access each structured output separately
print("CODE:")
print(result.content.code)
print("\nEXPLANATION:")
print(result.content.explanation)
print("\nCOMPLEXITY:")
print(result.content.complexity)
```

## Prompt Template Usage

The `CustomizeAgent` also supports using `PromptTemplate` for more flexible prompt templating. For detailed information about prompt templates and their advanced features, please refer to the [PromptTemplate Tutorial](./prompt_template.md).

### Simple Prompt Template

Here's a basic example using a prompt template:

```python
from evoagentx.prompts import StringTemplate

agent = CustomizeAgent(
    name="FirstAgent",
    description="A simple agent that prints hello world",
    prompt_template=StringTemplate(
        instruction="Print 'hello world'",
    ),
    llm_config=openai_config
)

message = agent()
print(message.content.content)
```

### Prompt Template with Inputs and Outputs

You can combine prompt templates with structured inputs and outputs:

```python
code_writer = CustomizeAgent(
    name="CodeWriter",
    description="Writes Python code based on requirements",
    prompt_template=StringTemplate(
        instruction="Write Python code that implements the provided `requirement`",
        # You can optionally add demonstrations:
        # demonstrations=[
        #     {
        #         "requirement": "print 'hello world'",
        #         "code": "print('hello world')"
        #     }, 
        #     {
        #         "requirement": "print 'Test Demonstration'",
        #         "code": "print('Test Demonstration')"
        #     }
        # ]
    ), # no need to specify input placeholders in the instruction of the prompt template
    llm_config=openai_config,
    inputs=[
        {"name": "requirement", "type": "string", "description": "The coding requirement"}
    ],
    outputs=[
        {"name": "code", "type": "string", "description": "The generated Python code"},
    ],
    parse_mode="custom", 
    parse_func=lambda content: {"code": extract_code_blocks(content)[0]}
)

message = code_writer(
    inputs={"requirement": "Write a function that returns the sum of two numbers"}
)
print(message.content.code)
```

The `PromptTemplate` provides a more structured way to define prompts and can include:
- A main instruction
- Optional context that can be used to provide additional information
- Optional constraints that the LLM should follow 
- Optional demonstrations for few-shot learning
- Optional tools information that the LLM can use 
etc. 

!!! note
    1. When using `prompt_template`, you don't need to explicitly include input placeholders in the instruction string like `{input_name}`. The template will automatically handle the mapping of inputs. 

    2. Also, you don't need to explicitly specify the output format in the `instruction` field of the `PromptTemplate`. The template will automatically formulate the output format based on the `outputs` parameter and the `parse_mode` parameter. However, `PromptTemplate` also supports explicitly specifying the output format by specifying `PromptTemplate.format(custom_output_format="...")`. 


## Parsing Modes

CustomizeAgent supports different ways to parse the LLM output:

### 1. String Mode (`parse_mode="str"`)

Uses the raw LLM output as the value for each output field. Useful for simple agents where structured parsing isn't needed.

```python
agent = CustomizeAgent(
    name="SimpleAgent",
    description="Returns raw output",
    prompt="Generate a greeting for {name}",
    inputs=[{"name": "name", "type": "string", "description": "The name to greet"}],
    outputs=[{"name": "greeting", "type": "string", "description": "The generated greeting"}],
    parse_mode="str",
    # other parameters...
)
```

After executing the agent, you can access the raw LLM response by `message.content.content` or `message.content.greeting`.  

### 2. Title Mode (`parse_mode="title"`, default)

Extracts content between titles matching output field names. This is the default parsing mode.

```python
agent = CustomizeAgent(
    name="ReportGenerator",
    description="Generates a structured report",
    prompt="Create a report about {topic}",
    outputs=[
        {"name": "summary", "type": "string", "description": "Brief summary"},
        {"name": "analysis", "type": "string", "description": "Detailed analysis"}
    ],
    # Default title pattern is "## {title}"
    title_format="### {title}",  # Optional: customize title format
    # other parameters...
)
```
With this configuration, the LLM should be instructed to format its response like (only required when passing the complete prompt with `prompt` parameter to instantiate the `CustomizeAgent`. If using `prompt_template`, you don't need to specify this):

```
### summary
Brief summary of the topic here.

### analysis
Detailed analysis of the topic here.
```

!!! note
    The section titles output by the LLM should be exactly the same as the output field names. Otherwise, the parsing will fail. For instance, in above example, if the LLM outputs `### Analysis`, which is different from the output field name `analysis`, the parsing will fail. 

### 3. JSON Mode (`parse_mode="json"`)

Parse the JSON string output by the LLM. The keys of the JSON string should be exactly the same as the output field names. 

```python
agent = CustomizeAgent(
    name="DataExtractor",
    description="Extracts structured data",
    prompt="Extract key information from this text: {text}",
    inputs=[
        {"name": "text", "type": "string", "description": "The text to extract information from"}
    ],
    outputs=[
        {"name": "people", "type": "string", "description": "Names of people mentioned"},
        {"name": "places", "type": "string", "description": "Locations mentioned"},
        {"name": "dates", "type": "string", "description": "Dates mentioned"}
    ],
    parse_mode="json",
    # other parameters...
)
```
When using this mode, the LLM should output a valid JSON string with keys matching the output field names. For instance, you should instruct the LLM to output (only required when passing the complete prompt with `prompt` parameter to instantiate the `CustomizeAgent`. If using `prompt_template`, you don't need to specify this):

```json
{
    "people": "extracted people",
    "places": "extracted places",
    "dates": "extracted dates"
}
```
If there are multiple JSON string in the LLM response, only the first one will be used. 

### 4. XML Mode (`parse_mode="xml"`)

Parse the XML string output by the LLM. The keys of the XML string should be exactly the same as the output field names.  

```python
agent = CustomizeAgent(
    name="DataExtractor",
    description="Extracts structured data",
    prompt="Extract key information from this text: {text}",
    inputs=[
        {"name": "text", "type": "string", "description": "The text to extract information from"}
    ],
    outputs=[
        {"name": "people", "type": "string", "description": "Names of people mentioned"},
    ],
    parse_mode="xml",
    # other parameters...
)
```

When using this mode, the LLM should generte texts containing xml tags with keys matching the output field names. For instance, you should instruct the LLM to output (only required when passing the complete prompt with `prompt` parameter to instantiate the `CustomizeAgent`. If using `prompt_template`, you don't need to specify this):

```xml
The people mentioned in the text are: <people>John Doe and Jane Smith</people>.
```

If the LLM output contains multiple xml tags with the same name, only the first one will be used. 

### 5. Custom Parsing (`parse_mode="custom"`)

For maximum flexibility, you can define a custom parsing function:

```python
from evoagentx.core.registry import register_parse_function

@register_parse_function  # Register the function for serialization
def extract_python_code(content: str) -> dict:
    """Extract Python code from LLM response"""
    code_blocks = extract_code_blocks(content)
    return {"code": code_blocks[0] if code_blocks else ""}

agent = CustomizeAgent(
    name="CodeExplainer",
    description="Generates and explains code",
    prompt="Write a Python function that {requirement}",
    inputs=[
        {"name": "requirement", "type": "string", "description": "The requirement to generate code for"}
    ],
    outputs=[
        {"name": "code", "type": "string", "description": "The generated code"},
    ],
    parse_mode="custom",
    parse_func=extract_python_code,
    # other parameters...
)
```

!!! note 
    1. The parsing function should have an input parameter `content` that takes the raw LLM response as input, and return a dictionary with keys matching the output field names. 

    2. It is recommended to use the `@register_parse_function` decorator to register the parsing function for serialization, so that you can save the agent and load it later. 


## Saving and Loading Agents

You can save agent definitions to reuse them later:

```python
# Save agent configuration. By default, the `llm_config` will not be saved. 
code_writer.save_module("./agents/code_writer.json")

# Load agent from file (requires providing llm_config again)
loaded_agent = CustomizeAgent.from_file(
    "./agents/code_writer.json", 
    llm_config=openai_config
)
```

## Advanced Example: Multi-Step Code Generator

Here's a more advanced example that demonstrates creating a specialized code generation agent with multiple structured outputs:

```python
from pydantic import Field
from evoagentx.actions import ActionOutput
from evoagentx.core.registry import register_parse_function

class CodeGeneratorOutput(ActionOutput):
    code: str = Field(description="The generated Python code")
    documentation: str = Field(description="Documentation for the code")
    tests: str = Field(description="Unit tests for the code")

@register_parse_function
def parse_code_documentation_tests(content: str) -> dict:
    """Parse LLM output into code, documentation, and tests sections"""
    sections = content.split("## ")
    result = {"code": "", "documentation": "", "tests": ""}
    
    for section in sections:
        if not section.strip():
            continue
        
        lines = section.strip().split("\n")
        section_name = lines[0].lower()
        section_content = "\n".join(lines[1:]).strip()
        
        if "code" in section_name:
            # Extract code from code blocks
            code_blocks = extract_code_blocks(section_content)
            result["code"] = code_blocks[0] if code_blocks else section_content
        elif "documentation" in section_name:
            result["documentation"] = section_content
        elif "test" in section_name:
            # Extract code from code blocks if present
            code_blocks = extract_code_blocks(section_content)
            result["tests"] = code_blocks[0] if code_blocks else section_content
    
    return result

# Create the advanced code generator agent
advanced_generator = CustomizeAgent(
    name="AdvancedCodeGenerator",
    description="Generates complete code packages with documentation and tests",
    prompt="""
    Create a complete implementation based on this requirement:
    {requirement}
    
    Provide your response in the following format:
    
    ## Code
    [Include the Python code implementation here]
    
    ## Documentation
    [Include clear documentation explaining the code]
    
    ## Tests
    [Include unit tests that verify the code works correctly]
    """,
    llm_config=openai_config,
    inputs=[
        {"name": "requirement", "type": "string", "description": "The coding requirement"}
    ],
    outputs=[
        {"name": "code", "type": "string", "description": "The generated Python code"},
        {"name": "documentation", "type": "string", "description": "Documentation for the code"},
        {"name": "tests", "type": "string", "description": "Unit tests for the code"}
    ],
    output_parser=CodeGeneratorOutput,
    parse_mode="custom",
    parse_func=parse_code_documentation_tests,
    system_prompt="You are an expert Python developer specialized in writing clean, efficient code with comprehensive documentation and tests."
)

# Execute the agent
result = advanced_generator(
    inputs={
        "requirement": "Create a function to validate if a string is a valid email address"
    }
)

# Access the structured outputs
print("CODE:")
print(result.content.code)
print("\nDOCUMENTATION:")
print(result.content.documentation)
print("\nTESTS:")
print(result.content.tests)
```

This advanced example demonstrates how to create a specialized agent that produces multiple structured outputs from a single LLM call, providing a complete code package with implementation, documentation, and tests.
# Workflow Graph

## Introduction

The `WorkFlowGraph` class is a fundamental component in the EvoAgentX framework for creating, managing, and executing complex AI agent workflows. It provides a structured way to define task dependencies, execution order, and the flow of data between tasks.

A workflow graph represents a collection of tasks (nodes) and their dependencies (edges) that need to be executed in a specific order to achieve a goal. The `SequentialWorkFlowGraph` is a specialized implementation that focuses on linear workflows with a single path from start to end.

## Architecture

### WorkFlowGraph Architecture

A `WorkFlowGraph` consists of several key components:

1. **Nodes (WorkFlowNode)**: 
   
    Each node represents a task or operation in the workflow, with the following properties:

    - `name`: A unique identifier for the task
    - `description`: Detailed description of what the task does
    - `inputs`: List of input parameters required by the task, each input parameter is an instance of `Parameter` class. 
    - `outputs`: List of output parameters produced by the task, each output parameter is an instance of `Parameter` class. 
    - `agents` (optional): List of agents that can execute this task, each agent should be a **string** that matches the name of the agent in the `agent_manager` or a **dictionary** that specifies the agent name and its configuration, which will be used to create a `CustomizeAgent` instance in the `agent_manager`.  Please refer to the [Customize Agent](./customize_agent.md) documentation for more details about the agent configuration. 
    - `action_graph` (optional): An instance of `ActionGraph` class, where each action is an instance of the `Operator` class. Please refer to the [Action Graph](./action_graph.md) documentation for more details about the action graph. 
    - `status`: Current execution state of the task (PENDING, RUNNING, COMPLETED, FAILED).

    !!! note 
        1. You should provide either `agents` or `action_graph` to execute the task. If both are provided, `action_graph` will be used. 

        2. If you provide a set of `agents`, these agents will work together to complete the task. When executing the task using `WorkFlow`, the system will automatically determine the execution sequence (actions) based on the agent information and execution history. Specifically, when executing the task, `WorkFlow` will analyze all the possible actions within these agents and repeatly select the best action to execute based on the task description and execution history. 

        3. If you provide an `action_graph`, it will be directly used to complete the task. When executing the task with `WorkFlow`, the system will execute the actions in the order defined by the `action_graph` and return the results.  


2. **Edges (WorkFlowEdge)**: 
   
    Edges represent dependencies between tasks, defining execution order and data flow. Each edge has:

    - `source`: Name of the source node (where the edge starts)
    - `target`: Name of the target node (where the edge ends) 
    - `priority` (optional): numeric priority to influence execution order

3. **Graph Structure**:
   
    Internally, the workflow is represented as a directed graph where:

    - Nodes represent tasks
    - Edges represent dependencies and data flow between tasks
    - The graph structure supports both linear sequences and more complex patterns:
        - Fork-join patterns (parallel execution paths that rejoin later)
        - Conditional branches
        - Potential cycles (loops) in the workflow

4. **Node States**:
   
    Each node in the workflow can be in one of the following states:
    
    - `PENDING`: The task is waiting to be executed
    - `RUNNING`: The task is currently being executed
    - `COMPLETED`: The task has been successfully executed
    - `FAILED`: The task execution has failed

### SequentialWorkFlowGraph Architecture

The `SequentialWorkFlowGraph` is a specialized implementation of `WorkFlowGraph` that automatically infers node connections to create a linear workflow. It's designed for simpler use cases where tasks need to be executed in sequence, with outputs from one task feeding into the next.

#### Input Format

The `SequentialWorkFlowGraph` accepts a simplified input format that makes it easy to define linear workflows. Instead of explicitly defining nodes and edges, you provide a list of tasks in the order they should be executed. Each task is defined as a dictionary with the following fields:

- `name` (required): A unique identifier for the task
- `description` (required): Detailed description of what the task does
- `inputs` (required): List of input parameters for the task
- `outputs` (required): List of output parameters produced by the task
- `prompt` (required): The prompt template to guide the agent's behavior
- `system_prompt` (optional): System message to provide context to the agent
- `output_parser` (optional): The output parser to parse the output of the task 
- `parse_mode` (optional): Mode for parsing outputs, defaults to "str"
- `parse_func` (optional): Custom function for parsing outputs
- `parse_title` (optional): Title for the parsed output

The parameters related to prompts and parsing will be used to create a `CustomizeAgent` instance in the `agent_manager`. Please refer to the [Customize Agent](./customize_agent.md) documentation for more details about the agent configuration. 

#### Internal Conversion to WorkFlowGraph

Internally, `SequentialWorkFlowGraph` automatically converts this simplified task list into a complete `WorkFlowGraph` by:

1. **Creating WorkFlowNode instances**: For each task in the input list, it creates a `WorkFlowNode` with appropriate properties. During this process:

    - It converts the task definition into a node with inputs, outputs, and an associated agent.
    - It automatically generates a unique agent name based on the task name.
    - It configures the agent with the provided prompt, system_prompt, and parsing options.

2. **Inferring edge connections**: It examines the input and output parameters of each task and automatically creates `WorkFlowEdge` instances to connect tasks where outputs from one task match the inputs of another.

3. **Building the graph structure**: Finally, it constructs the complete directed graph representing the workflow, with all nodes and edges properly connected.

This automatic conversion process makes it significantly easier to define sequential workflows without needing to manually specify all the graph components.

## Usage

### Basic WorkFlowGraph Creation & Execution 

```python
from evoagentx.workflow.workflow_graph import WorkFlowNode, WorkFlowGraph, WorkFlowEdge
from evoagentx.workflow.workflow import WorkFlow 
from evoagentx.agents import AgentManager, CustomizeAgent 
from evoagentx.models import OpenAILLMConfig, OpenAILLM 

llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="xxx", stream=True, output_response=True)
llm = OpenAILLM(llm_config)

agent_manager = AgentManager()

data_extraction_agent = CustomizeAgent(
    name="DataExtractionAgent",
    description="Extract data from source",
    inputs=[{"name": "data_source", "type": "string", "description": "Source data location"}],
    outputs=[{"name": "extracted_data", "type": "string", "description": "Extracted data"}],
    prompt="Extract data from source: {data_source}",
    llm_config=llm_config
)  

data_transformation_agent = CustomizeAgent(
    name="DataTransformationAgent",
    description="Transform data",
    inputs=[{"name": "extracted_data", "type": "string", "description": "Extracted data"}],
    outputs=[{"name": "transformed_data", "type": "string", "description": "Transformed data"}],
    prompt="Transform data: {extracted_data}",
    llm_config=llm_config
)

# add agents to the agent manager for workflow execution 
data_extraction_agent = agent_manager.add_agents(agents = [data_extraction_agent, data_transformation_agent])

# Create workflow nodes
task1 = WorkFlowNode(
    name="Task1",
    description="Extract data from source",
    inputs=[{"name": "data_source", "type": "string", "description": "Source data location"}],
    outputs=[{"name": "extracted_data", "type": "string", "description": "Extracted data"}],
    agents=["DataExtractionAgent"] # should match the name of the agent in the agent manager
)

task2 = WorkFlowNode(
    name="Task2",
    description="Transform data",
    inputs=[{"name": "extracted_data", "type": "string", "description": "Data to transform"}],
    outputs=[{"name": "transformed_data", "type": "string", "description": "Transformed data"}],
    agents=["DataTransformationAgent"] # should match the name of the agent in the agent manager
)

task3 = WorkFlowNode(
    name="Task3",
    description="Analyze data and generate insights",
    inputs=[{"name": "transformed_data", "type": "string", "description": "Data to analyze"}],
    outputs=[{"name": "insights", "type": "string", "description": "Generated insights"}],
    agents=[
        {
            "name": "DataAnalysisAgent",
            "description": "Analyze data and generate insights",
            "inputs": [{"name": "transformed_data", "type": "string", "description": "Data to analyze"}],
            "outputs": [{"name": "insights", "type": "string", "description": "Generated insights"}],
            "prompt": "Analyze data and generate insights: {transformed_data}",
            "parse_mode": "str",
        } # will be used to create a `CustomizeAgent` instance in the `agent_manager`
    ]
)

# Create workflow edges
edge1 = WorkFlowEdge(source="Task1", target="Task2")
edge2 = WorkFlowEdge(source="Task2", target="Task3")

# Create the workflow graph
workflow_graph = WorkFlowGraph(
    goal="Extract, transform, and analyze data to generate insights",
    nodes=[task1, task2, task3],
    edges=[edge1, edge2]
)

# add agents to the agent manager for workflow execution 
agent_manager.add_agents_from_workflow(workflow_graph, llm_config=llm_config)

# create a workflow instance for execution 
workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
workflow.execute(inputs={"data_source": "xxx"})
```

### Creating a SequentialWorkFlowGraph

```python
from evoagentx.workflow.workflow_graph import SequentialWorkFlowGraph

# Define tasks with their inputs, outputs, and prompts
tasks = [
    {
        "name": "DataExtraction",
        "description": "Extract data from the specified source",
        "inputs": [
            {"name": "data_source", "type": "string", "required": True, "description": "Source data location"}
        ],
        "outputs": [
            {"name": "extracted_data", "type": "string", "required": True, "description": "Extracted data"}
        ],
        "prompt": "Extract data from the following source: {data_source}", 
        "parse_mode": "str"
    },
    {
        "name": "DataTransformation",
        "description": "Transform the extracted data",
        "inputs": [
            {"name": "extracted_data", "type": "string", "required": True, "description": "Data to transform"}
        ],
        "outputs": [
            {"name": "transformed_data", "type": "string", "required": True, "description": "Transformed data"}
        ],
        "prompt": "Transform the following data: {extracted_data}", 
        "parse_mode": "str"
    },
    {
        "name": "DataAnalysis",
        "description": "Analyze data and generate insights",
        "inputs": [
            {"name": "transformed_data", "type": "string", "required": True, "description": "Data to analyze"}
        ],
        "outputs": [
            {"name": "insights", "type": "string", "required": True, "description": "Generated insights"}
        ],
        "prompt": "Analyze the following data and generate insights: {transformed_data}", 
        "parse_mode": "str"
    }
]

# Create the sequential workflow graph
sequential_workflow_graph = SequentialWorkFlowGraph(
    goal="Extract, transform, and analyze data to generate insights",
    tasks=tasks
)
```

### Saving and Loading a Workflow

```python
# Save workflow
workflow_graph.save_module("examples/output/my_workflow.json")

# For SequentialWorkFlowGraph, use save_module and get_graph_info
sequential_workflow_graph.save_module("examples/output/my_sequential_workflow.json")
```

### Visualizing the Workflow

```python
# Display the workflow graph with node statuses visually
workflow_graph.display()
```

The `WorkFlowGraph` and `SequentialWorkFlowGraph` classes provide a flexible and powerful way to design complex agent workflows, track their execution, and manage the flow of data between tasks. 
# Action Graph

## Introduction

The `ActionGraph` class is a fundamental component in the EvoAgentX framework for creating and executing sequences of operations (actions) within a single task. It provides a structured way to define, manage, and execute a series of operations that need to be performed in a specific order to complete a task.

An action graph represents a collection of operators (actions) that are executed in a predefined sequence to process inputs and produce outputs. Unlike the `WorkFlowGraph` which manages multiple tasks and their dependencies at a higher level, the `ActionGraph` focuses on the detailed execution steps within a single task.

## Architecture

### ActionGraph Architecture

An `ActionGraph` consists of several key components:

1. **Operators**: 
   
    Each operator represents a specific operation or action that can be performed as part of a task, with the following properties:

    - `name`: A unique identifier for the operator
    - `description`: Detailed description of what the operator does
    - `llm`: The LLM used to execute the operator
    - `outputs_format`: The structured format of the output of the operator
    - `interface`: The interface for calling the operator.
    - `prompt`: Template used to guide the LLM when executing this operator

2. **LLM**: 
   
    The ActionGraph uses a Language Learning Model (LLM) to execute the operators. It receives a `llm_config` as input and create an LLM instance, which will be passed to the operators for execution. The LLM provides the reasoning and generation capabilities needed to perform each action.

3. **Execution Flow**:
   
    The ActionGraph defines a specific execution sequence:

    - Actions are executed in a predetermined order (specified in the `execute` or `async_execute` method using code)
    - Each action can use the results from previous actions
    - The final output is produced after all actions have been executed

### Comparison with WorkFlowGraph

While both `ActionGraph` and `WorkFlowGraph` manage execution flows, they operate at different levels of abstraction:

| Feature | ActionGraph | WorkFlowGraph |
|---------|-------------|---------------|
| Scope | Single task execution | Multi-task workflow orchestration |
| Components | Operators (actions) | Nodes (tasks) and edges (dependencies) |
| Focus | Detailed steps within a task | Relationships between different tasks |
| Flexibility | Fixed execution sequence | Dynamic execution based on dependencies |
| Primary use | Define reusable task execution patterns | Orchestrate complex multi-step workflows |
| Granularity | Fine-grained operations | Coarse-grained tasks |

## Usage

### Basic ActionGraph Creation

```python
from evoagentx.workflow import ActionGraph
from evoagentx.workflow.operators import Custom
from evoagentx.models import OpenAILLMConfig 

# Create LLM configuration
llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="xxx")

# Create a custom ActionGraph
class MyActionGraph(ActionGraph):
    def __init__(self, llm_config, **kwargs):

        name = kwargs.pop("name") if "name" in kwargs else "Custom Action Graph"
        description = kwargs.pop("description") if "description" in kwargs else "A custom action graph for text processing"
        # create an LLM instance `self._llm` based on the `llm_config` and pass it to the operators
        super().__init__(name=name, description=description, llm_config=llm_config, **kwargs)
        # Define operators
        self.extract_entities = Custom(self._llm) # , prompt="Extract key entities from the following text: {input}")
        self.analyze_sentiment = Custom(self._llm) # , prompt="Analyze the sentiment of the following text: {input}")
        self.summarize = Custom(self._llm) # , prompt="Summarize the following text in one paragraph: {input}")

    def execute(self, text: str) -> dict:
        # Execute operators in sequence (specify the execution order of operators)
        entities = self.extract_entities(input=text, instruction="Extract key entities from the provided text")["response"]
        sentiment = self.analyze_sentiment(input=text, instruction="Analyze the sentiment of the provided text")["response"]
        summary = self.summarize(input=text, instruction="Summarize the provided text in one paragraph")["response"]

        # Return combined results
        return {
            "entities": entities,
            "sentiment": sentiment,
            "summary": summary
        }

# Create the action graph
action_graph = MyActionGraph(llm_config=llm_config)

# Execute the action graph
result = action_graph.execute(text="This is a test text")
print(result)
```

### Using ActionGraph in WorkFlowGraph

You can either use `ActionGraph` directly or use it in `WorkFlowGraph` as a node. 

```python
from evoagentx.workflow.workflow_graph import WorkFlowNode, WorkFlowGraph
from evoagentx.workflow.action_graph import QAActionGraph
from evoagentx.core.base_config import Parameter
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.workflow import WorkFlow

# Create LLM configuration
llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="xxx", stream=True, output_response=True)
llm = OpenAILLM(llm_config)

# Create an action graph
qa_graph = QAActionGraph(llm_config=llm_config)

# Create a workflow node that uses the action graph
qa_node = WorkFlowNode(
    name="QATask",
    description="Answer questions using a QA system",
    # input names should match the parameters in the `execute` method of the action graph
    inputs=[Parameter(name="problem", type="string", description="The problem to answer")],
    outputs=[Parameter(name="answer", type="string", "description": "The answer to the problem")],
    action_graph=qa_graph  # Using action_graph instead of agents
)

# Create the workflow graph
workflow_graph = WorkFlowGraph(goal="Answer a question", nodes=[qa_node])

# define the workflow 
workflow = WorkFlow(graph=workflow_graph, llm=llm)

# Execute the workflow
result = workflow.execute(inputs={"problem": "What is the capital of France?"})
print(result)
```

!!! warning 
    When using `ActionGraph` in `WorkFlowNode`, the `inputs` parameter of the `WorkFlowNode` should match the required parameters in the `execute` method of the `ActionGraph`. The `execute` method is expected to return a **dictionary** or `LLMOutputParser` instance with keys matching the names of the `outputs` in the `WorkFlowNode`. 

### Saving and Loading an ActionGraph

```python
# Save action graph
action_graph.save_module("examples/output/my_action_graph.json")

# Load action graph
from evoagentx.workflow.action_graph import ActionGraph
loaded_graph = ActionGraph.from_file("examples/output/my_action_graph.json", llm_config=llm_config)
```

The `ActionGraph` class provides a powerful way to define complex sequences of operations within a single task, complementing the higher-level orchestration capabilities of the `WorkFlowGraph` in the EvoAgentX framework.
# Build Your First Agent

In EvoAgentX, agents are intelligent components designed to complete specific tasks autonomously. This tutorial will walk you through the essential concepts of creating and using agents in EvoAgentX:

1. **Creating a Simple Agent with CustomizeAgent**: Learn how to create a basic agent with custom prompts 
2. **Working with Multiple Actions**: Create more complex agents that can perform multiple tasks
3. **Saving and Loading Agents**: Learn how to save and load your agents

By the end of this tutorial, you'll be able to create both simple and complex agents, understand how they process inputs and outputs, and know how to save and reuse them in your projects.

## 1. Creating a Simple Agent with CustomizeAgent

The easiest way to create an agent is using `CustomizeAgent`, which allows you to quickly define an agent with a specific prompt.  

First, let's import the necessary components and setup the LLM:

```python
import os 
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig
from evoagentx.agents import CustomizeAgent

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure LLM
openai_config = OpenAILLMConfig(
    model="gpt-4o-mini", 
    openai_key=OPENAI_API_KEY, 
    stream=True
)
``` 

Now, let's create a simple agent that prints hello world. There are two ways to create a CustomizeAgent:

### Method 1: Direct Initialization
You can directly initialize the agent with the `CustomizeAgent` class: 
```python
first_agent = CustomizeAgent(
    name="FirstAgent",
    description="A simple agent that prints hello world",
    prompt="Print 'hello world'", 
    llm_config=openai_config # specify the LLM configuration 
)
```

### Method 2: Creating from Dictionary

You can also create an agent by defining its configuration in a dictionary:

```python
agent_data = {
    "name": "FirstAgent",
    "description": "A simple agent that prints hello world",
    "prompt": "Print 'hello world'",
    "llm_config": openai_config
}
first_agent = CustomizeAgent.from_dict(agent_data) # use .from_dict() to create an agent. 
```

### Using the Agent

Once created, you can use the agent to print hello world. 

```python
# Execute the agent without input. The agent will return a Message object containing the results. 
message = first_agent()

print(f"Response from {first_agent.name}:")
print(message.content.content) # the content of a Message object is a LLMOutputParser object, where the `content` attribute is the raw LLM output. 
```

For a complete example, please refer to the [CustomizeAgent example](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/customize_agent.py). 

CustomizeAgent also offers other features including structured inputs/outputs and multiple parsing strategies. For detailed information, see the [CustomizeAgent documentation](../modules/customize_agent.md).

## 2. Creating an Agent with Multiple Actions

In EvoAgentX, you can create an agent with multiple predefined actions. This allows you to build more complex agents that can perform multiple tasks. Here's an example showing how to create an agent with `TestCodeGeneration` and `TestCodeReview` actions:

### Defining Actions
First, we need to define the actions, which are subclasses of `Action`: 
```python
from evoagentx.agents import Agent
from evoagentx.actions import Action, ActionInput, ActionOutput

# Define the CodeGeneration action inputs
class TestCodeGenerationInput(ActionInput):
    requirement: str = Field(description="The requirement for the code generation")

# Define the CodeGeneration action outputs
class TestCodeGenerationOutput(ActionOutput):
    code: str = Field(description="The generated code")

# Define the CodeGeneration action
class TestCodeGeneration(Action): 

    def __init__(
        self, 
        name: str="TestCodeGeneration", 
        description: str="Generate code based on requirements", 
        prompt: str="Generate code based on requirements: {requirement}",
        inputs_format: ActionInput=None, 
        outputs_format: ActionOutput=None, 
        **kwargs
    ):
        inputs_format = inputs_format or TestCodeGenerationInput
        outputs_format = outputs_format or TestCodeGenerationOutput
        super().__init__(
            name=name, 
            description=description, 
            prompt=prompt, 
            inputs_format=inputs_format, 
            outputs_format=outputs_format, 
            **kwargs
        )
    
    def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, **kwargs) -> TestCodeGenerationOutput:
        action_input_attrs = self.inputs_format.get_attrs() # obtain the attributes of the action input 
        action_input_data = {attr: inputs.get(attr, "undefined") for attr in action_input_attrs}
        prompt = self.prompt.format(**action_input_data) # format the prompt with the action input data 
        output = llm.generate(
            prompt=prompt, 
            system_message=sys_msg, 
            parser=self.outputs_format, 
            parse_mode="str" # specify how to parse the output 
        )
        if return_prompt:
            return output, prompt
        return output


# Define the CodeReview action inputs
class TestCodeReviewInput(ActionInput):
    code: str = Field(description="The code to be reviewed")
    requirements: str = Field(description="The requirements for the code review")

# Define the CodeReview action outputs
class TestCodeReviewOutput(ActionOutput):
    review: str = Field(description="The review of the code")

# Define the CodeReview action
class TestCodeReview(Action):
    def __init__(
        self, 
        name: str="TestCodeReview", 
        description: str="Review the code based on requirements", 
        prompt: str="Review the following code based on the requirements:\n\nRequirements: {requirements}\n\nCode:\n{code}.\n\nYou should output a JSON object with the following format:\n```json\n{{\n'review': '...'\n}}\n```", 
        inputs_format: ActionInput=None, 
        outputs_format: ActionOutput=None, 
        **kwargs
    ):
        inputs_format = inputs_format or TestCodeReviewInput
        outputs_format = outputs_format or TestCodeReviewOutput
        super().__init__(
            name=name, 
            description=description, 
            prompt=prompt, 
            inputs_format=inputs_format, 
            outputs_format=outputs_format, 
            **kwargs
        )
    
    def execute(self, llm: Optional[BaseLLM] = None, inputs: Optional[dict] = None, sys_msg: Optional[str]=None, return_prompt: bool = False, **kwargs) -> TestCodeReviewOutput:
        action_input_attrs = self.inputs_format.get_attrs()
        action_input_data = {attr: inputs.get(attr, "undefined") for attr in action_input_attrs}
        prompt = self.prompt.format(**action_input_data)
        output = llm.generate(
            prompt=prompt, 
            system_message=sys_msg,
            parser=self.outputs_format, 
            parse_mode="json" # specify how to parse the output 
        ) 
        if return_prompt:
            return output, prompt
        return output
```

From the above example, we can see that in order to define an action, we need to:

1. Define the action inputs and outputs using `ActionInput` and `ActionOutput` classes
2. Create an action class that inherits from `Action`
3. Implement the `execute` method which formulates the prompt with the action input data and uses the LLM to generate output, and specify how to parse the output using `parse_mode`.

### Defining an Agent 

Once we have defined the actions, we can create an agent by adding the actions to it:

```python
# Initialize the LLM
openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=os.getenv("OPENAI_API_KEY"))

# Define the agent 
developer = Agent(
    name="Developer", 
    description="A developer who can write code and review code",
    actions=[TestCodeGeneration(), TestCodeReview()], 
    llm_config=openai_config
)
```

### Executing Different Actions

Once you've created an agent with multiple actions, you can execute specific actions:

```python
# List all available actions on the agent
actions = developer.get_all_actions()
print(f"Available actions of agent {developer.name}:")
for action in actions:
    print(f"- {action.name}: {action.description}")

# Generate some code using the CodeGeneration action
generation_result = developer.execute(
    action_name="TestCodeGeneration", # specify the action name
    action_input_data={ 
        "requirement": "Write a function that returns the sum of two numbers"
    }
)

# Access the generated code
generated_code = generation_result.content.code
print("Generated code:")
print(generated_code)

# Review the generated code using the CodeReview action
review_result = developer.execute(
    action_name="TestCodeReview",
    action_input_data={
        "requirements": "Write a function that returns the sum of two numbers",
        "code": generated_code
    }
)

# Access the review results
review = review_result.content.review
print("\nReview:")
print(review)
```

This example demonstrates how to:
1. List all available actions on an agent
2. Generate code using the TestCodeGeneration action
3. Review the generated code using the TestCodeReview action
4. Access the results from each action execution

For a complete working example, please refer to the [Agent example](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/agent_with_multiple_actions.py). 


## 3. Saving and Loading Agents

You can save an agent to a file and load it later:

```python
# Save the agent to a file
developer.save_module("examples/output