# Agents Groups Buiding Library Documentation

## Overview

This library provides abstractions and utilities for creating, managing, and orchestrating AI agents using **LangGraph** , **LangChain**, and other libraries that will be added with time. It allows developers to define agents, group them into workflows, manage their tools and prompts, and control agent states.

Key capabilities include:

* Defining **LangGraph agents** with structured prompts and toolkits.
* Grouping agents into **agent groups** for orchestrated workflows.
* Managing **tools** for agent tasks.
* Creating reusable **prompt templates**.
* Tracking **agent states** via typed dictionaries.
* Simplifying agent and group creation with **factory classes**.

The library is ideal for developers familiar with LangChain/LangGraph who want a higher-level abstraction for agent orchestration.

---

## Installation

> Installation instructions are pending. Include PyPI or GitHub link when available.

```bash
# Example placeholder
pip install langgraph-agent-lib
```

---

## Quick Start Example

```python
from your_library import (
    PromptTemplate,
    ToolKit,
    BaseStatus,
    AgentsFactory,
    GroupsFactory
)

# Create a prompt template
prompt = PromptTemplate(
    role="assistant",
    question="Summarize this text",
    context="Some context here",
    language="en",
    constraints="Keep under 100 words",
    output="summary",
    rules="Always be concise"
)

# Create a toolkit and register a tool
toolkit = ToolKit()
toolkit.register(lambda text: text.upper())

# Create initial status
status = BaseStatus(prompt="Hello Agent")

# Create agent using factory
agent_factory = AgentsFactory()
agent = agent_factory.create_lang_graph_agent(
    prompt=prompt,
    input_state="prompt",
    next_state="completed",
    tools=toolkit,
    llm=None,
    agent_type="langgraph",
    verbose=True
)

# Create a group using factory
group_factory = GroupsFactory()
group = group_factory.create_lang_graph_group(status=status)

# Sign agent to group and define entry point
group.sign_agent(name="Summarizer", agent=agent)
group.sign_entry_point(agent_name="Summarizer")

# Run the group
result = group.run(prompt="Process this text")
print(result)
```

---

## Entity Documentation

### Agent

#### **LangGraphAgent**

Represents an AI agent that executes tasks using LangGraph and a set of tools.
It encapsulates a prompt template, input/output state management, and a toolkit for executing functions.

**Constructor:**

```python
class LangGraphAgent(IAgent):
    def __init__(
        self,
        prompt: PromptTemplate,
        input_state: str,
        next_state: str,
        tools: IToolKit,
        llm,
        agent_type,
        verbose: bool
    ):
        ...
```

**Parameters:**

* `prompt` (`PromptTemplate`): The structured prompt defining the agent’s task.
* `input_state` (`str`): Name of the state from which the agent reads input.
* `next_state` (`str`): State name to set after processing.
* `tools` (`IToolKit`): Collection of callable tools for the agent.
* `llm` (`Any`): Language model used by the agent (can be LangChain LLM).
* `agent_type` (`Any`): Type of agent (currently LangGraph).
* `verbose` (`bool`): Enables detailed logging.

**Methods:**

* `prepare(state: Dict[str, Any]) -> Dict[str, Any]`
  Prepares the agent for execution based on the current state.

  ```python
  prepared_state = agent.prepare({"prompt": "Hello"})
  ```

* `build_agent()`
  Builds the internal LangGraph agent instance ready for execution.

**Interactions:**

* Uses `PromptTemplate` to generate task instructions.
* Uses `ToolKit` to access registered tools for execution.
* Works within a **LangGraphGroup** for orchestrated workflows.

---

### Agent Groups

#### **LangGraphGroup**

Represents a collection of agents working together. Handles orchestration, input/output flow, and agent dependencies.

**Constructor:**

```python
class LangGraphGroup(IGroup):
    def __init__(self, status: BaseStatus):
        ...
```

**Parameters:**

* `status` (`BaseStatus`): Initial agent state for the group.

**Methods:**

* `sign_agent(name: str, agent: IAgent)`
  Registers an agent under a unique name.

* `sign_edge(from_agent: str, to_agent: str)`
  Connects output of one agent to another agent as input.

* `sign_entry_point(agent_name: str)`
  Defines which agent starts execution when `run()` is called.

* `get_agent(name: str) -> IAgent`
  Returns the agent object for a given name.

* `run(prompt: str) -> Any`
  Executes the group starting from the entry point and processes the given prompt.

**Interactions:**

* Orchestrates multiple `LangGraphAgent` instances.
* Uses agent output states to trigger next agents.
* Supports tool usage indirectly through agents.

---

### ToolKit

#### **ToolKit**

Stores and manages callable tools that agents can execute.

**Constructor:**

```python
class ToolKit(IToolKit):
    def __init__(self):
        ...
```

**Methods:**

* `register(tool: Callable)`
  Adds a new callable tool to the toolkit.

* `get_all_tools()`
  Returns a list of all registered tools.

**Interactions:**

* Used by agents to perform actions or transformations.
* Tools are registered independently from agents and reused across them.

---

### PromptTemplate

Generates structured prompts for agents.

**Constructor:**

```python
class PromptTemplate:
    def __init__(self, role, question, context, language, constraints, output, rules):
        ...
```

**Parameters:**

* `role` (`str`): Role of the agent (e.g., "assistant").
* `question` (`str`): The task or query for the agent.
* `context` (`str`): Additional context to guide the agent.
* `language` (`str`): Language of the prompt.
* `constraints` (`str`): Rules or limitations for agent response.
* `output` (`str`): Expected output type or format.
* `rules` (`str`): Behavioral rules for the agent.

**Methods:**

* `get_command() -> str`
  Returns the formatted prompt string for the agent.

---

### Agent Status

#### **BaseStatus**

A typed dictionary to store agent prompt state.

```python
class BaseStatus(TypedDict):
    prompt: str
```

* `prompt` (`str`): Current agent input or context.

---

### Factories

#### AgentsFactory

Simplifies agent creation.

**Method:**

```python
def create_lang_graph_agent(...) -> LangGraphAgent:
    ...
```

#### GroupsFactory

Simplifies group creation.

**Method:**

```python
def create_lang_graph_group(status: BaseStatus) -> LangGraphGroup:
    ...
```

---

## Component Interactions

* **Agent ↔ PromptTemplate**: Agents use `PromptTemplate` to format input for processing.
* **Agent ↔ ToolKit**: Agents execute registered tools for tasks.
* **Group ↔ Agent**: Groups orchestrate agents, manage execution order, and handle state transitions.
* **Factory Classes**: Provide standardized creation patterns for agents and groups, reducing boilerplate.

---

## Best Practices & Notes

* Always register agents in a group before defining edges or entry points.
* Use `ToolKit` to centralize commonly used tools for reusability.
* Prefer `PromptTemplate` for structured, repeatable prompts.
* Ensure `input_state` and `next_state` are correctly mapped to avoid state conflicts.
* Verbose mode in agents is useful for debugging workflows.

---

## Example Workflow

```python
# Define tools
toolkit = ToolKit()
toolkit.register(lambda x: x[::-1])  # Reverse text

# Define prompt
prompt = PromptTemplate(
    role="assistant",
    question="Reverse the text",
    context="None",
    language="en",
    constraints="Return only reversed string",
    output="text",
    rules="Do not modify input other than reversing"
)

# Create agent
agent_factory = AgentsFactory()
agent = agent_factory.create_lang_graph_agent(
    prompt=prompt,
    input_state="prompt",
    next_state="done",
    tools=toolkit,
    llm=None,
    agent_type="langgraph",
    verbose=False
)

# Create group
status = class AgentStatus(BaseStatus):
    prompt: str="Hello"
    done: str

group_factory = GroupsFactory()
group = group_factory.create_lang_graph_group(status=status)
group.sign_agent("Reverser", agent)
group.sign_entry_point("Reverser")

# Run
result = group.run(prompt="Hello World")
print(result)  # Expected output: "dlroW olleH"
```

---

*End of Documentation.*
