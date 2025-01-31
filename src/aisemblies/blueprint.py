from collections import deque
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Any,
    Callable,
)

from .utils import import_function


@dataclass
class StationConfig:
    """
    Holds all configuration and runtime information about a station in the assembly line.


    Attributes
    ----------
    name : str
        The unique station name by which it can be referenced in the Blueprint.
    import_path : str | None
        A dotted import path to dynamically load the station function (e.g. "my_package.my_module.my_station").
    func : Callable[..., Any] | None
        A direct Python callable to run if no import_path is given. If both import_path and func are
        provided, func takes precedence.
    transitions : dict[str, str]
        A mapping from station output to the name of the next station to run.
    finish_on : list[Any]
        A list of outputs for which the assembly line should finish execution.
        If the station output is in this list, the line ends immediately.
    on_error : str | None
        The name of a station to jump to if this station's function raises an unhandled exception.
        If not set, the assembly line will raise an error.
    """

    name: str
    import_path: str | None = field(default=None, repr=False)
    func: Callable[..., Any] | None = field(default=None, repr=False)
    transitions: dict[str, str] = field(default_factory=dict)
    finish_on: list[Any] = field(default_factory=list)
    on_error: str | None = field(default=None)

    def __post_init__(self):
        """
        Validate input consistency and ensure we have at least
        one way to obtain the callable (import_path or func).
        """
        if (self.func is None) and (not self.import_path):
            raise ValueError(
                f"Station '{self.name}' must have either an 'import_path' or a 'func' defined."
            )
        if (self.func is not None) and (not callable(self.func)):
            raise TypeError(
                f"'func' provided for station '{self.name}' is not callable."
            )

    @property
    def function(self) -> Callable[..., Any]:
        """
        Return the actual Python callable associated with this station.

        If the station was defined with a direct `func`, use it.
        Otherwise, we attempt to import via the provided `import_path`.

        Raises
        ------
        TypeError
            If the imported object is not a callable.

        Returns
        -------
        Callable[..., Any]
            The actual function to be executed.
        """
        if self.func is not None:
            return self.func
        if self.import_path:
            module_name, func_name = self.import_path.rsplit(".", 1)
            return import_function(module_name, func_name)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert this station config into a dict structure that
        might be helpful for exporting or for hooking into your
        existing AssemblyLine.

        Returns
        -------
        dict[str, Any]
        """
        return {
            "function": self.import_path
            or (self.func.__module__ + "." + self.func.__name__),
            "transitions": self.transitions,
            "finish_on": self.finish_on,
            "on_error": self.on_error,
        }


class Blueprint:
    """
    A manager for the entire assembly line configuration.


    Attributes
    ----------
    _stations : dict[str, StationConfig]
        Internal dictionary mapping station names to their station configs.
    _entry_station : str | None
        The station name where the assembly line should begin.
    _global_error_station : str | None
        The default error station name to jump to if no local station error handler is defined.
        This can be None (in which case we rely on the AssemblyLine's fallback error handling).
    """

    def __init__(self) -> None:
        self._stations: dict[str, StationConfig] = {}
        self._entry_station: str | None = None
        self._global_error_station: str | None = None

    def add_station(
        self,
        name: str,
        *,
        import_path: str | None = None,
        func: Callable[..., Any] | None = None,
        transitions: dict[str, str] | None = None,
        finish_on: list[Any] | None = None,
        on_error: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Register a new station in this blueprint. If a station with the same name
        already exists and 'overwrite' is False, an error will be raised.

        Parameters
        ----------
        name : str
            The name of the station.
        import_path : str | None, default=None
            Dotted path to import the station function dynamically.
        func : Callable[..., Any] | None, default=None
            Direct callable for this station (takes precedence over import_path).
        transitions : dict[str, str] | None, default=None
            Mapping from station output -> next station name.
        finish_on : list[Any] | None, default=None
            A list of outputs that should cause the assembly line to finish.
        on_error : str | None, default=None
            Station name to jump to if an unhandled error occurs in this station.
        overwrite : bool, default=False
            If False, adding a station name that already exists raises an error.

        Raises
        ------
        ValueError
            If the station name already exists and overwrite=False.
        """
        if (name in self._stations) and not overwrite:
            raise ValueError(
                f"Station '{name}' is already defined. Use overwrite=True to replace it."
            )

        station = StationConfig(
            name=name,
            import_path=import_path,
            func=func,
            transitions=transitions or {},
            finish_on=finish_on or [],
            on_error=on_error,
        )
        self._stations[name] = station

    def stations_transitioning_to(self, target_st: str) -> list[str]:
        """
        Return a list of station names that directly transition to 'target_st'.

        Parameters
        ----------
        target_st : str
            The station name we are checking for direct transitions to.

        Returns
        -------
        list[str]
            A list of station names from which 'target_st' is reachable in a single step.

        Raises
        ------
        KeyError
            If `target_st` is not a defined station in this blueprint.
        """
        if target_st not in self._stations:
            raise KeyError(f"Station '{target_st}' does not exist in this blueprint.")

        result = []
        for st, st_cfg in self._stations.items():
            for _, next_st in st_cfg.transitions.items():
                if next_st == target_st:
                    result.append(st)
        return result

    def stations_referencing(self, target_st: str) -> list[str]:
        """
        Return a list of all station names that eventually lead to 'target_st',
        directly or indirectly.

        This includes any station that can transition, via a chain of transitions,
        to 'target_st'.

        Parameters
        ----------
        target_st : str
            The name of the station to be reached.

        Returns
        -------
        list[str]
            A list of station names for which there exists a path of transitions leading
            to 'target_st'. The 'target_st' is excluded from this list.

        Raises
        ------
        KeyError
            If `target_st` is not a defined station in this blueprint.
        """
        if target_st not in self._stations:
            raise KeyError(f"Station '{target_st}' does not exist in this blueprint.")

        reversed_adj = {name: set() for name in self._stations}
        for st, st_cfg in self._stations.items():
            for _, next_st in st_cfg.transitions.items():
                reversed_adj[next_st].add(st)

        visited = set([target_st])
        queue = deque([target_st])
        while queue:
            current = queue.popleft()
            for predecessor in reversed_adj[current]:
                if predecessor not in visited:
                    visited.add(predecessor)
                    queue.append(predecessor)

        visited.remove(target_st)
        return list(visited)

    def set_entry_station(self, st: str) -> None:
        """
        Specify which station the assembly line should begin with.

        Parameters
        ----------
        st : str
            The name of an already-added station.

        Raises
        ------
        ValueError
            If the station being set is not found in this blueprint.
        """
        if st not in self._stations:
            raise ValueError(f"Station '{st}' is not defined in the blueprint.")
        self._entry_station = st

    def set_global_error_station(self, st: str) -> None:
        """
        Specify a 'global fallback' station to jump to if a station raises an error
        but does not define a local 'on_error'.

        Raises
        ------
        ValueError
            If the station is not defined.
        """
        if st not in self._stations:
            raise ValueError(f"Station '{st}' is not defined in the blueprint.")
        self._global_error_station = st

    @property
    def entry_station(self) -> str:
        """
        Return the entry station name for this blueprint.

        Raises
        ------
        ValueError
            If no entry station has been set.

        Returns
        -------
        str
        """
        if self._entry_station is None:
            raise ValueError(
                "Entry station is not set. Call set_entry_station() first."
            )
        return self._entry_station

    @property
    def global_error_station(self) -> str | None:
        """
        Return the globally configured error station name, or None if not set.

        Returns
        -------
        str | None
            The name of the global error station, or None if none is configured.
        """
        return self._global_error_station

    def get_station_config(self, st: str) -> StationConfig:
        """
        Retrieve the StationConfig object for the given station.

        Parameters
        ----------
        st : str
            Station name to look up.

        Raises
        ------
        KeyError
            If the station does not exist.

        Returns
        -------
        StationConfig
        """
        try:
            return self._stations[st]
        except KeyError as exc:
            raise KeyError(f"Station '{st}' does not exist in this blueprint.") from exc

    def get_station_callable(self, st: str) -> Callable[..., Any]:
        """
        Retrieve the Python callable that this station points to (either from func or an import_path).

        Parameters
        ----------
        st : str
            The station name.

        Returns
        -------
        Callable[..., Any]
        """
        station_config = self.get_station_config(st)
        return station_config.function

    @property
    def stations(self) -> MappingProxyType:
        """
        Return a read-only view of the current station registry. This is
        analogous to an immutable stations dictionary.

        Returns
        -------
        MappingProxyType
        """
        return MappingProxyType(self._stations)

    def __repr__(self) -> str:
        lines = []
        entry = self._entry_station or "None"
        global_err = self._global_error_station or "None"
        for name, cfg in self._stations.items():
            func_import_path = cfg.import_path
            if not func_import_path and cfg.func:
                func_import_path = f"{cfg.func.__module__}.{cfg.func.__name__}"
            lines.append(
                f"\tStation '{name}':\n"
                f"\t\tfunction='{func_import_path}'\n"
                f"\t\ttransitions={cfg.transitions}\n"
                f"\t\tfinish_on={cfg.finish_on}\n"
                f"\t\ton_error={cfg.on_error}"
            )
        return (
            f"Blueprint(\n"
            f"\tentry_station='{entry}',\n"
            f"\tglobal_error_station='{global_err}',\n" + "\n".join(lines) + "\n)"
        )

    def __str__(self) -> str:
        lines = []
        entry = self._entry_station or "None"
        global_err = self._global_error_station or "None"
        for name, cfg in self._stations.items():
            if cfg.import_path:
                func_name = cfg.import_path.rsplit(".", 1)[-1]
            else:
                func_name = cfg.func.__name__ if cfg.func else "???"
            lines.append(
                f"\tStation '{name}':\n"
                f"\t\tfunction_name='{func_name}'\n"
                f"\t\ttransitions={cfg.transitions}\n"
                f"\t\tfinish_on={cfg.finish_on}\n"
                f"\t\ton_error={cfg.on_error}"
            )
        return (
            f"Blueprint:(\n"
            f"\tentry_station='{entry}',\n"
            f"\tglobal_error_station='{global_err}',\n" + "\n".join(lines) + "\n)"
        )
