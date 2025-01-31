# AISemblies

AISemblies is a small, experimental Python library for orchestrating multi-agent (or multi-function) pipelines using a finite-state machine–inspired model. It borrows ideas from assembly lines and state machines to structure and run your pipeline asynchronously, with clear transitions and error-handling logic. If you’ve found existing solutions like [langgraph](https://github.com/langchain-ai/langgraph) too heavyweight with many abstractions or opinionated, or you’re curious about a more minimal, “build it yourself” approach to multi-agent orchestration, AISemblies might be interesting to explore.  

> Note: This is not meant to replace langgraph or any other libraries. It’s not nearly as feature-complete or polished as tools like [langgraph](https://github.com/langchain-ai/langgraph), [autogen](https://github.com/microsoft/autogen), or [Pydantic AI](https://github.com/pydantic/pydantic-ai). Rather, it’s a personal experiment in creating a simpler, flexible foundation—hoping to keep a "do one thing well" ethos: finite-state multi-function orchestration.

--------------------------------------------------------------------------------

## Table of Contents

- [Motivation](#motivation)
- [Features](#features)
- [Key Concepts](#key-concepts)
  - [Blueprint](#blueprint)
  - [AssemblyLine](#assemblyline)
  - [Stations](#stations)
  - [Station Transitions & Error Handling](#station-transitions--error-handling)
- [Installation](#installation)
- [Quick Start Example](#quick-start-example)
  - [1. Create the Blueprint](#1-create-the-blueprint)
  - [2. Run the Assembly Line](#2-run-the-assembly-line)
  - [3. Querying the Example Server](#3-querying-the-example-server)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
  - [Serialization](#serialization)
  - [Running Multiple Loads Concurrently](#running-multiple-loads-concurrently)
  - [Callbacks and Streaming](#callbacks-and-streaming)
- [Roadmap & Limitations](#roadmap--limitations)
- [License](#license)
- [Contributing & Feedback](#contributing--feedback)
- [Acknowledgments](#acknowledgments)

--------------------------------------------------------------------------------

## Motivation

1. **Easier “graph” or “pipeline” serialization.** AISemblies stores the station names and import paths, transitions, and (optionally) error-handling stations in a JSON/YAML blueprint. You can import that blueprint in another environment—provided the actual Python callables are also accessible—to reconstruct and run the exact same pipeline.  
2. **Customizable streaming/logging.** Instead of forcing you into a single streaming model, AISemblies delegates all streaming/logging to user-defined station functions. You can display partial results in real time or integrate your favorite logging approach, right in those station callables.  
3. **Arbitrary pipeline state.** The “running load” is just any Python object or data structure. AISemblies doesn’t impose fixed schema constraints. You’re free to mutate or replace this “running load” inside your station functions, or keep it immutable and pass updates along—whatever suits your use case.  

--------------------------------------------------------------------------------

## Features

- **Finite-State Machine**: Each station function is a node in your pipeline graph. It returns an output that informs AISemblies which station to run next (or to stop).  
- **Error Handling (local + global)**: Define a local `on_error` handler for any station or a global fallback error station. If a station fails and no local error handler is defined, AISemblies checks if you’ve defined a global error station; otherwise it raises an `AssemblyLineError`.  
- **Async Execution**: If your station calls are async, you can dispatch multiple data loads concurrently.
- **JSON/YAML Serialization**: Export your `Blueprint` as JSON or YAML, import it elsewhere, and reconstruct the same pipeline. (You do need to have the actual station functions importable by dotted path or directly in the environment.)  
- **Lightweight**: Minimal core functionality (see `core.py`, `blueprint.py`, `serialization.py`). You can wrap or integrate with your existing logging or streaming libraries, or attach sophisticated instrumentation as callback wrappers.

--------------------------------------------------------------------------------

## Key Concepts

### Blueprint

A `Blueprint` is effectively a declarative “map” of your pipeline:

- Holds all `StationConfig` objects (one per station).
- Defines station transitions (outputs → the name of the next station).
- Tracks `entry_station` (where the assembly starts).
- Optionally defines a `global_error_station` for error fallback.

### AssemblyLine

An `AssemblyLine` orchestrates pipeline execution using the given `Blueprint`. You can pass an initial payload (`running_load`) into `run_one_load_async`. The pipeline proceeds station by station (each station is an async Python function).  

When a station finishes (returns a value), AISemblies checks which station to run next via the `transitions` mapping. If the result value is in a station’s `finish_on` list or no next station is found, the pipeline stops.

### Stations

Each station is represented by a `StationConfig`, which has:

- A Python function (callable) or a dotted `import_path`.
- A dictionary of `transitions` from possible return values → next station name.
- Optional `finish_on` list indicating outputs that terminate the pipeline. 
- Optional `on_error` station name for local error handling.

### Station Transitions & Error Handling

Station results drive the pipeline. For instance, in the snippet below:

```python
crag_blueprint.add_station(
    name="retrieve",
    func=retrieve,
    transitions={"OK": "grade_documents", "NO_QUESTION": None},
    on_error="error_handler",
)
```

- If `retrieve(...)` returns `"OK"`, the pipeline proceeds to `grade_documents`.  
- If it returns `"NO_QUESTION"`, we have `None` which indicates no next station—the pipeline ends.  
- If `retrieve(...)` raises an exception, AISemblies jumps to the station named `"error_handler"` (since we specified `on_error="error_handler"`).

--------------------------------------------------------------------------------

## Installation

AISemblies is not (yet) published to PyPI. To install from source:

1. Clone the repository or download the source.  
2. Navigate to the cloned directory.  
3. If you want to run the example you have to install both the `openai` and `examples` extras. You might also need to set environment variables for LLM providers (like OpenAI) or for specialized search tools (e.g., [TavilySearch](https://github.com/langchain-ai/langchain-community)).

--------------------------------------------------------------------------------

## Quick Start Example

The repo includes an example called “CRAG” (Corrective RAG), inspired by the [langgraph CRAG Cookbook example](https://github.com/langchain-ai/langchain/blob/master/cookbook/langgraph_crag.ipynb). The pipeline attempts to retrieve documents, grade them for relevance, transform the query for web search if needed, do further retrieval, and finally generate an answer. It’s just a demonstration of how to define stations, transitions, and error handling within AISemblies.

### 1. Create the Blueprint

Inside [`crag.py`](./src/aisemblies/examples/create_bp.py), we set up:

- Station `"retrieve"` to retrieve from a vector store,  
- Station `"grade_documents"` to decide if we have relevant docs or need to transform the query,  
- And so on, ultimately leading to station `"generate"`, or to `"error_handler"` on failure.

Inside `create_bp.py` we create create and save the blueprint as a YAML file.

Run:

```bash
uv run src/aisemblies/examples/create_bp.py
```

It'll produce the file `crag_blueprint.yaml`, which describes the pipeline’s stations and transitions.

### 2. Run the Assembly Line

For demonstration, there is a minimal FastAPI app in [`app/app.py`](./src/aisemblies/examples/app/app.py). It:

- Loads the `crag_blueprint.yaml` via `blueprint_from_yaml()`.
- Constructs an `AssemblyLine` from that Blueprint.
- Defines a `/crag` endpoint to run the pipeline for a single question.
- Defines `/crag_many` to run multiple questions concurrently, streaming partial results.

You can launch it with:

```bash
uv run src/aisemblies/examples/app/app.py
```

### 3. Querying the Example Server

Inside [`query_app.py`](./src/aisemblies/examples/query_app.py), you’ll find code that posts questions to the server:

```bash
uv run src/aisemblies/examples/query_app.py
```

- It calls `/crag_many` with a list of questions, then streams line-by-line results.  
- You’ll see each partial response from the pipeline as it finishes retrieval, search, or generation.
- You'll also see colored outputs to your server's console.

--------------------------------------------------------------------------------

## Project Structure

```
.
├── LICENSE
├── pyproject.toml
├── README.md
├── src
│   └── aisemblies
│       ├── blueprint.py        # Core of station config & blueprint classes
│       ├── core.py             # The AssemblyLine orchestration
│       ├── examples/           # Example: CRAG pipeline
│       │   ├── app/            # Minimal FastAPI server
│       │   ├── crag_blueprint.yaml (blueprint as a YAML file)
│       │   ├── create_bp.py    # Create the blueprint YAML file
│       │   ├── crag.py         # Station functions for the CRAG flow
│       │   └── query_app.py    # Query the server
│       ├── exceptions.py       # Generic excepton
│       ├── messages.py         # Utilities to send messages to OpenAI clients
│       ├── responses.py        # Utilities to parse responses from OpenAI clients
│       ├── serialization.py    # JSON/YAML import/export for blueprints
│       ├── tools.py            # Utilities for using tool calling with OpenAI clients 
│       └── utils.py
└── uv.lock
```

--------------------------------------------------------------------------------

## How It Works

### Serialization

The main serialization logic lives in [`serialization.py`](./src/aisemblies/serialization.py). When you call `blueprint_to_yaml(bp, filepath)` or `blueprint_to_json(bp, filepath)`, AISemblies exports:

- **Station names** and references to either a function’s dotted path or a fallback "module.func" if no explicit path was given.  
- **Transitions**: what output goes to which station.  
- **`on_error`** station references.  
- **Entry station** and optional **global error station**.

Deserializing with `blueprint_from_yaml(filepath)` (or `_json`) re-creates the Blueprint object in memory. If your stations rely on local Python callables, ensure they’re importable in the environment.

### Running Multiple Loads Concurrently

Use `AssemblyLine.run_many_loads_async(list_of_loads, callback=...)` to run multiple initial data payloads concurrently. AISemblies spawns tasks for each load in parallel, culminating in a single shared event loop. If you pass a `callback`, it is invoked each time a load finishes.

### Callbacks and Streaming

AISemblies doesn’t enforce any single streaming approach. In the CRAG example (`crag.py`), we show how to manually handle streamed tokens from an OpenAI-compatible client to the server's console by writing partial tokens. You could also pass in your own logging or streaming callbacks.

This user-driven approach means you can do rich logging, partial content display, or real-time processing just by customizing your station’s function body. AISemblies doesn’t intercept or wrap your stream logic—your station is free to yield partial outputs or update the console as you see fit.

--------------------------------------------------------------------------------

## Roadmap & Limitations

1. **Performance**: For extremely large or complex pipelines (100+ stations), you may want to optimize how transitions and adjacency are stored. Also, the concurrency part could be improved.  
2. **More Natural Streaming**: Potential improvements include letting stations yield partial results while continuing to the next station or standard callbacks for streaming.  
3. **Improved Logging**: You might want structured logging, logs hooking to an external aggregator, etc. Right now, you do it within your station functions. A more unified approach is under consideration.  
4. **Circular/Looped Graph**: Since AISemblies is basically a state-machine handler, it can handle transitions that form cycles but you should be careful with infinite loops.  

--------------------------------------------------------------------------------

## License

This project is licensed under the terms described in the [LICENSE](./LICENSE) file included in this repository.

--------------------------------------------------------------------------------

## Contributing & Feedback

Feel free to open a PR, an issue or a discussion if you have bug fixes, improvements, or new ideas and suggestions. This is a personal exploration of a lighter-weight pipeline approach, so design critiques (architecture, naming, error-handling approach, etc.) are very welcome.

--------------------------------------------------------------------------------

## Acknowledgments

- The “CRAG” example is inspired by [langgraph’s CRAG Cookbook example](https://github.com/langchain-ai/langchain/blob/master/cookbook/langgraph_crag.ipynb).   
- The approach to station transitions is reminiscent of the [transitions](https://github.com/pytransitions/transitions) library and finite state machines.


**Thank you for reading and trying out AISemblies!** If you have suggestions or feedback, please open an issue or PR.