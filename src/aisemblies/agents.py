import asyncio
import importlib
import inspect
import json
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Protocol, Set, TypeVar

from pydantic import BaseModel, Field

from aisemblies.messages import (
    AssistantMessage,
    AssistantResponse,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from aisemblies.tool_helpers import invoke_llm_tool_calls
from aisemblies.tools import FunctionTool, ToolCollection


class DataclassProtocol(Protocol):
    __dataclass_fields__: dict


TDataclass = TypeVar("TDataclass", bound=DataclassProtocol)


class DataTransit:
    """
    Holds the input and output for each agent's step in the assembly.
    agent_id -> list of inputs or outputs
    Because we can have multiple parents, each agent might store a single list of inputs.
    """

    def __init__(self):
        self.inputs: Dict[str, List[Any]] = defaultdict(list)
        self.intermediary: Dict[str, List[Any]] = defaultdict(list)
        self.outputs: Dict[str, Any] = {}

    def record_input(self, agent_id: str, data: Any):
        self.inputs[agent_id].append(data)

    def record_output(self, agent_id: str, data: Any):
        self.outputs[agent_id] = data

    def record_intermediary(self, agent_id: str, data: Any):
        self.intermediary[agent_id].append(data)


class Agent(ABC):
    """
    An abstract Agent with a single async `run()` method.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    @abstractmethod
    async def run(
        self,
        data_transit: DataTransit,
        input_data: List[BaseModel | dict | TDataclass],
    ) -> Any:
        """
        Each agent receives a list of inputs from its parents. The agent can decide
        how to merge or handle them. The agent then returns (and records) a single output object.
        """
        ...


class FunctionToolConfig(BaseModel):
    """
    Holds enough information to dynamically reconstruct a tool (function).
    """

    import_path: str
    func_name: str
    description: str | None = None


class LLMConfig(BaseModel):
    model_name: str = "gpt-4"
    system_prompt: str | None = None
    user_prompt: str | None = None
    api_key_alias: str | None = None
    max_tool_call_iters: int = 3
    tool_configs: list[FunctionToolConfig] = Field(default_factory=list)

    def build_tool_collection(self) -> ToolCollection | None:
        """
        Reconstruct a ToolCollection from self.tool_configs.
        """
        tool_objs = []
        for cfg in self.tool_configs:
            fn = _import_function(cfg.import_path, cfg.func_name)
            tool_objs.append(
                FunctionTool(
                    func=fn,
                    name=cfg.func_name,
                    description=cfg.description,
                )
            )
        return ToolCollection(tool_objs) if tool_objs else None


class LLMAgent(Agent):
    """
    An Agent that calls an LLM (OpenAI's /chat/completions).
    """

    def __init__(self, agent_id: str, config: LLMConfig, client):
        super().__init__(agent_id)
        self.config = config
        self.client = client
        self.tools = self.config.build_tool_collection()

    async def run(
        self,
        data_transit: DataTransit,
        input_data: List[BaseModel | dict | TDataclass],
    ) -> AssistantResponse:
        """
        1) Record the (multiple) inputs
        2) Build the system + user messages
        3) Call the OpenAI API
        4) Parse the response
        5) Record and return the output
        """
        for item in input_data:
            data_transit.record_input(self.agent_id, item)

        messages = []

        if self.config.system_prompt:
            system_msg = SystemMessage(self.config.system_prompt)
            messages.append(system_msg.to_msg())

        if self.config.user_prompt:
            user_msg = UserMessage(self.config.user_prompt)
            for item in input_data:
                user_msg = user_msg.render(item)
            messages.append(user_msg.to_msg())

        if not messages:
            messages.append(UserMessage("").to_msg())

        final_response = await self._run_llm_with_tools(
            messages=messages,
            tools=self.tools,
            max_iters=self.config.max_tool_call_iters,
            data_transit=data_transit,
        )

        data_transit.record_output(self.agent_id, final_response)
        return final_response

    async def _run_llm_with_tools(
        self,
        messages: list[dict],
        tools: ToolCollection,
        max_iters: int,
        data_transit: DataTransit,
    ) -> AssistantResponse:
        """
        Repeatedly calls the LLM with an evolving conversation:
         - Check if the LLM wants to call a tool
         - If yes, run the tool, append the tool's output as a "tool message"
         - Then call the LLM again
         - Stop if no more tools are requested or we exceed max_iters
        """
        iters = 0
        while iters < max_iters:
            completion = await self._call_llm_once(messages, tools)
            parsed_completion = AssistantResponse.from_completion(completion)

            print("COMPLETION")
            print(parsed_completion.first_choice)

            all_calls = parsed_completion.all_tool_calls
            if not all_calls:
                return parsed_completion

            assistant_msg = AssistantMessage.from_response(parsed_completion)
            messages.append(assistant_msg.to_msg())

            tool_calls = invoke_llm_tool_calls(
                parsed_completion,
                tools,
                raise_on_unknown_tool=True,
                choice_idx=0,
            )
            data_transit.record_intermediary(self.agent_id, tool_calls)
            tool_messages = []
            for result in tool_calls:
                msg = ToolMessage(
                    content=str(result["output"]),
                    tool_call_id=result["call_id"],
                )
                tool_messages.append(msg)
            messages.extend(tool_messages)

            data_transit.record_intermediary(self.agent_id, tool_calls)
            iters += 1

    async def _call_llm_once(
        self, messages: list[dict], tools: ToolCollection
    ) -> Any:
        """
        Minimal async wrapper around e.g. openai.ChatCompletion.acreate().
        Pass `tools=...` or their OpenAI JSON if your environment supports function calling.
        """
        if self.client is None:
            raise ValueError(
                f"LLMAgent {self.agent_id} has no client. "
                "Provide an api_key_alias or manually assign a client."
            )

        print("MESSAGES")
        print(messages)

        if tools:
            openai_tools = tools.to_openai_list()
            completion = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
            )
        else:
            completion = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
            )

        return completion


class FunctionConfig(BaseModel):
    import_path: str | None = None
    func_name: str | None = None
    docstring: str | None = None
    signature: str | None = None
    return_type: str | None = None
    func: Callable[..., Any] | None = None


def _import_function(module_name: str, func_name: str) -> Callable[..., Any]:
    """
    Dynamically import a function by module_name and func_name.
    """
    mod = importlib.import_module(module_name)
    fn = getattr(mod, func_name)
    return fn


class FunctionAgent(Agent):
    """
    An Agent that calls a Python function with the input data (list).
    The function can be sync or async. We'll detect and await it if needed.
    """

    def __init__(self, agent_id: str, config: FunctionConfig):
        super().__init__(agent_id)
        self.config = config

        if (
            not callable(self.config.func)
            and self.config.import_path
            and self.config.func_name
        ):
            self.config.func = _import_function(
                self.config.import_path, self.config.func_name
            )

    async def run(
        self,
        data_transit: DataTransit,
        input_data: List[BaseModel | dict | TDataclass],
    ) -> Any:
        for item in input_data:
            data_transit.record_input(self.agent_id, item)

        fn = self.config.func
        if fn is None:
            raise ValueError(
                f"FunctionAgent {self.agent_id}: config.func is None."
            )

        if asyncio.iscoroutinefunction(fn):
            result = await fn(input_data)
        else:
            result = fn(input_data)

        data_transit.record_output(self.agent_id, result)
        return result


def _get_function_details(fn: Callable[..., Any]) -> FunctionConfig:
    """
    Inspect a function and return a FunctionConfig with docstring, signature, etc.
    Also capture the return type if available.
    """
    mod_name = fn.__module__
    func_name = fn.__name__
    import_path = mod_name
    docstring = inspect.getdoc(fn) or ""
    sig = inspect.signature(fn)
    signature_str = str(sig)

    return_annotation = sig.return_annotation
    if return_annotation == inspect._empty:
        return_type = None
    else:
        return_type = str(return_annotation)

    return FunctionConfig(
        import_path=import_path,
        func_name=func_name,
        docstring=docstring,
        signature=signature_str,
        return_type=return_type,
        func=fn,
    )


def _load_api_key(api_key_alias: str, api_key_value_map: Dict[str, str]) -> str:
    """
    Load API key from the user-supplied map or from environment variables.
    If not found, raise an exception.
    """
    if api_key_alias in api_key_value_map:
        return api_key_value_map[api_key_alias]
    # fallback to environment
    if api_key_alias in os.environ:
        return os.environ[api_key_alias]
    raise ValueError(f"Could not find API key for alias '{api_key_alias}'.")


def _parse_prompt_fields(template: str | None) -> List[str] | None:
    """
    A simple approach to gather placeholders from a string like 'Given {foo} and {bar}' -> ['foo', 'bar'].
    Returns None if template is None or if no placeholders are found.
    """
    if not isinstance(template, str):
        return None

    fields = re.findall(r"\{(\w+)\}", template)
    return fields if fields else None


class DAGAssembly:
    """
    Orchestrates multiple agents in a directed acyclic graph (DAG).
    We use a topological scheduling approach, but run each agent as soon
    as all of its parents are done, i.e., continuously, not in waves.
    """

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        self.in_degree: Dict[str, int] = defaultdict(int)
        self.data_transit = DataTransit()

        self.parents_map: Dict[str, Set[str]] = defaultdict(set)

    def add_agents(self, *agents: Agent):
        """
        Add multiple agents at once.
        """
        for agent in agents:
            self.add_agent(agent)

    def add_agent(self, agent: Agent) -> None:
        """
        Add a single agent to the assembly, but do not define its edges yet.
        """
        if agent.agent_id in self.agents:
            raise ValueError(f"Agent ID {agent.agent_id} already exists.")
        self.agents[agent.agent_id] = agent
        if agent.agent_id not in self.in_degree:
            self.in_degree[agent.agent_id] = 0

    def add_connections(
        self, from_agent_ids: List[str], to_agent_ids: List[str]
    ) -> None:
        """
        Define that each agent in from_agent_ids feeds each agent in to_agent_ids.
        For example: from_agent_ids = [A, B], to_agent_ids = [C, D]
        means A->C, A->D, B->C, B->D
        """
        for f in from_agent_ids:
            for t in to_agent_ids:
                self.add_connection(f, t)

    def add_connection(self, from_agent_id: str, to_agent_id: str) -> None:
        """
        Single connection from from_agent_id to to_agent_id.
        """
        if from_agent_id not in self.agents:
            raise ValueError(f"Unknown agent: {from_agent_id}")
        if to_agent_id not in self.agents:
            raise ValueError(f"Unknown agent: {to_agent_id}")
        if to_agent_id not in self.graph[from_agent_id]:
            self.graph[from_agent_id].add(to_agent_id)
            self.parents_map[to_agent_id].add(from_agent_id)
            self.in_degree[to_agent_id] += 1

            self._validate_no_cycles()

    def _validate_no_cycles(self):
        """
        Perform a topological sort on a *copy* of the in_degree and graph data
        to ensure we can visit all agents. If not, there's a cycle.
        """
        in_degree_copy = dict(self.in_degree)

        queue = deque(
            [agent_id for agent_id, deg in in_degree_copy.items() if deg == 0]
        )

        visited_count = 0

        while queue:
            node = queue.popleft()
            visited_count += 1
            for succ in self.graph[node]:
                in_degree_copy[succ] -= 1
                if in_degree_copy[succ] == 0:
                    queue.append(succ)

        if visited_count < len(self.agents):
            raise ValueError(
                "Cycle detected in the agent assembly. The graph is not a valid DAG."
            )

    async def run(self, entry_inputs: Dict[str, BaseModel]) -> None:
        """
        Run the DAG.
        'entry_inputs' is a dict mapping agent_id -> user-provided data.
        For any agent with in_degree=0, we use these inputs from the user.
        Then, we schedule agents as soon as all their parents have finished.
        We *don't* store those inputs in data_transit ourselves;
        we'll pass them directly to _run_agent_when_ready().
        """
        tasks: Dict[str, asyncio.Task] = {}

        for agent_id, deg in self.in_degree.items():
            if deg == 0:
                user_data_for_agent = entry_inputs.get(agent_id)  # can be None
                tasks[agent_id] = asyncio.create_task(
                    self._run_agent_when_ready(
                        agent_id=agent_id,
                        tasks=tasks,
                        user_data=user_data_for_agent,
                    )
                )

        await asyncio.gather(*tasks.values())

    async def _run_agent_when_ready(
        self,
        agent_id: str,
        tasks: Dict[str, asyncio.Task],
        user_data: BaseModel | None = None,
    ):
        """
        Waits until all parent agents have finished, collects their outputs,
        merges them with any user_data (if this agent is an entry point),
        then calls agent.run().
        After finishing, it schedules its children if they are now ready.
        """
        for parent_id in self.parents_map[agent_id]:
            if parent_id in tasks:
                await tasks[parent_id]

        input_data_list: List[BaseModel | dict | TDataclass] = []
        for parent_id in self.parents_map[agent_id]:
            parent_output = self.data_transit.outputs.get(parent_id)
            if parent_output is not None:
                input_data_list.append(parent_output)

        if user_data is not None:
            input_data_list.append(user_data)

        agent = self.agents[agent_id]
        await agent.run(self.data_transit, input_data_list)

        for successor_id in self.graph[agent_id]:
            self.in_degree[successor_id] -= 1
            if self.in_degree[successor_id] == 0:
                tasks[successor_id] = asyncio.create_task(
                    self._run_agent_when_ready(
                        agent_id=successor_id, tasks=tasks, user_data=None
                    )
                )

    def export_to_json(self) -> str:
        """
        Returns a JSON string that describes:
          - All agents and their configurations
          - Their connections
          - Which ones are entry points (in_degree=0)
          - FunctionAgent details (import_path, func_name, docstring, signature)
          - LLMConfig details (api_key_alias, model_name, system_prompt, etc.)
          - Input schema info for entry-point agents if we can detect them
        """
        data = {
            "agents": [],
            "connections": [],
        }

        for agent_id, agent in self.agents.items():
            is_entry = self.in_degree[agent_id] == 0
            agent_info = {
                "agent_id": agent_id,
                "type": agent.__class__.__name__,
                "is_entry_point": is_entry,
            }

            if isinstance(agent, LLMAgent):
                agent_info["llm_config"] = agent.config.model_dump()
                sys_fields = _parse_prompt_fields(agent.config.system_prompt)
                user_fields = _parse_prompt_fields(agent.config.user_prompt)
                agent_info["system_prompt_fields"] = sys_fields
                agent_info["user_prompt_fields"] = user_fields

            elif isinstance(agent, FunctionAgent):
                fc = agent.config
                if fc.func and (not fc.import_path or not fc.func_name):
                    details = _get_function_details(fc.func)
                    fc.import_path = details.import_path
                    fc.func_name = details.func_name
                    fc.docstring = details.docstring
                    fc.signature = details.signature
                    fc.return_type = details.return_type

                agent_info["function_config"] = {
                    "import_path": fc.import_path,
                    "func_name": fc.func_name,
                    "docstring": fc.docstring,
                    "signature": fc.signature,
                    "return_type": fc.return_type,
                }

            data["agents"].append(agent_info)

        for from_agent, successors in self.graph.items():
            for to_agent in successors:
                data["connections"].append({"from": from_agent, "to": to_agent})

        return json.dumps(data, indent=2)

    @classmethod
    def from_json(
        cls, json_str: str, api_key_value_map: Dict[str, str] = None
    ) -> "DAGAssembly":
        """
        Reconstruct the DAGAssembly from JSON.
        For LLMAgents:
            - If an api_key_alias is present, we attempt to load the actual key from
              `api_key_value_map` or environment variables. Then we instantiate the LLM client.
        For FunctionAgents:
            - We dynamically import the function if 'import_path' and 'func_name' are given.
        If no api_key_alias is provided, we set client = None, so user must handle it manually.
        """
        if api_key_value_map is None:
            api_key_value_map = {}

        data = json.loads(json_str)
        assembly = cls()

        for a in data["agents"]:
            agent_id = a["agent_id"]
            agent_type = a["type"]

            if agent_type == "LLMAgent":
                llm_cfg_dict = a.get("llm_config", {})
                config_obj = LLMConfig(**llm_cfg_dict)

                client = None
                if config_obj.api_key_alias:
                    actual_key = _load_api_key(
                        config_obj.api_key_alias, api_key_value_map
                    )
                    from openai import AsyncOpenAI

                    client = AsyncOpenAI(api_key=actual_key)

                agent_obj = LLMAgent(
                    agent_id=agent_id,
                    config=config_obj,
                    client=client,
                )

            elif agent_type == "FunctionAgent":
                fconf_dict = a.get("function_config", {})
                fc = FunctionConfig(**fconf_dict)
                agent_obj = FunctionAgent(agent_id=agent_id, config=fc)

            else:
                raise ValueError(f"Unsupported agent type: {agent_type}")

            assembly.add_agent(agent_obj)

        for c in data["connections"]:
            assembly.add_connection(c["from"], c["to"])

        return assembly
