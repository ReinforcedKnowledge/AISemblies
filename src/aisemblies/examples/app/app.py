import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from rich.console import Console

from aisemblies.core import AssemblyLine
from aisemblies.serialization import blueprint_from_yaml


@asynccontextmanager
async def lifespan(app: FastAPI):
    console = Console()
    client = AsyncOpenAI()

    global_load = {
        "client": client,
        "console": console,
    }

    crag_blueprint = blueprint_from_yaml(
        "src/aisemblies/examples/crag_blueprint.yaml"
    )
    crag_assembly_line = AssemblyLine(blueprint=crag_blueprint)

    app.state.crag_blueprint = crag_blueprint
    app.state.crag_assembly_line = crag_assembly_line
    app.state.global_load = global_load

    yield


app = FastAPI(lifespan=lifespan)


@app.post("/crag")
async def run_crag(question: str = Body(..., embed=True)):
    load = {"question": question} | app.state.global_load
    assembly_line = app.state.crag_assembly_line
    await assembly_line.run_one_load_async(load)
    return {"result": load["generation"]}


@app.post("/crag_many")
async def run_crag_many(questions: list[str] = Body(...)):
    async def event_generator(questions: list[str]):
        loads = []
        for q in questions:
            load = {"question": q} | app.state.global_load
            loads.append(load)

        queue: asyncio.Queue = asyncio.Queue()

        def on_result_done(running_load: dict[str, Any]):
            question = running_load["question"]
            generation = running_load["generation"]
            data_dict = {
                "question": question,
                "answer": generation,
            }
            queue.put_nowait(json.dumps(data_dict))

        asyncio.create_task(
            app.state.crag_assembly_line.run_many_loads_async(
                loads, callback=on_result_done
            )
        )

        results_seen = 0
        while results_seen < len(questions):
            chunk = await queue.get()
            yield chunk + "\n"
            queue.task_done()
            results_seen += 1

    return StreamingResponse(
        event_generator(questions),
        media_type="application/json",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
