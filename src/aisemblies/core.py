import asyncio
import traceback
from typing import Any, Callable

from .blueprint import Blueprint
from .exceptions import AssemblyLineError


class AssemblyLine:
    """
    Orchestrates the execution of stations defined in a Blueprint using an
    asynchronous pipeline. It can process runs for a single load or multiple
    loads concurrently.

    This class encapsulates all logic for running an asynchronous assembly line.
    """

    def __init__(self, blueprint: Blueprint) -> None:
        """
        Initialize an AssemblyLine with a given Blueprint.

        Parameters
        ----------
        blueprint : Blueprint
            An instance of the Blueprint class that defines station configurations,
            transitions, and error handling behavior.
        """
        self._blueprint = blueprint

    async def _handle_unhandled_error(
        self,
        running_load: dict[str, Any],
        exception: Exception,
        traceback_str: str,
    ) -> None:
        """
        Default fallback error handler for unhandled exceptions.

        Parameters
        ----------
        running_load : dict[str, Any]
            The data being processed in the assembly line.
        exception : Exception
            The original exception that was raised.
        traceback_str : str
            A string representation of the traceback for debugging purposes.

        Raises
        ------
        AssemblyLineError
            Indicates that an unhandled asynchronous error occurred during station
            processing.
        """
        raise AssemblyLineError("Unhandled async error!") from exception

    async def run_one_load_async(
        self, initial_load: Any = None
    ) -> dict[str, Any]:
        """
        Process a single data load through the asynchronous assembly line.

        Parameters
        ----------
        initial_load : Any, optional
            The initial data payload to pass into the first station. Defaults to an
            empty dict if not provided.

        Returns
        -------
        dict[str, Any]
            A dictionary mapping station names to their station outputs. The assembly
            line terminates if a station output is in its `finish_on` list or if
            there is no valid next station.

        Raises
        ------
        AssemblyLineError
            If no local or global error handler is defined for an unhandled exception.
        """
        running_load = initial_load or {}
        station_outputs: dict[str, Any] = {}

        current_station = self._blueprint.entry_station
        while current_station:
            station_cfg = self._blueprint.get_station_config(current_station)
            station_func = station_cfg.function

            try:
                output = await station_func(running_load)
                station_outputs[current_station] = output
            except Exception as exc:
                tb_str = traceback.format_exc()
                err_station = (
                    station_cfg.on_error or self._blueprint.global_error_station
                )
                if err_station:
                    err_cfg = self._blueprint.get_station_config(err_station)
                    err_func = err_cfg.function
                    output = await err_func(
                        running_load, exception=exc, traceback_str=tb_str
                    )
                    station_outputs[err_station] = output
                else:
                    await self._handle_unhandled_error(
                        running_load, exc, tb_str
                    )
                    break

            if output in station_cfg.finish_on:
                break

            next_station = station_cfg.transitions.get(output)
            if not next_station:
                break
            current_station = next_station

        return running_load

    async def run_many_loads_async(
        self,
        loads: list[Any],
        callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Process multiple data loads concurrently through the asynchronous assembly line.

        Parameters
        ----------
        loads[Any]
            Each element in the list is an initial data payload passed to the pipeline.

        Returns
        -------
        list[dict[str, Any]]
            A list of station-output mappings—one dict per data load, in the same order.

        Raises
        ------
        AssemblyLineError
            If no local or global error handler is defined for an unhandled exception
            in any of the loads.
        """
        tasks = [self.run_one_load_async(load) for load in loads]
        results: list[dict[str, Any]] = []
        for coro in asyncio.as_completed(tasks):
            single_result = await coro
            results.append(single_result)
            if callback:
                callback(single_result)

        return results
