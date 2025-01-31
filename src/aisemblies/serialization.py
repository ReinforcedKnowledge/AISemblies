import json
from typing import Any

import yaml

from aisemblies.blueprint import Blueprint


def blueprint_to_dict(blueprint: Blueprint) -> dict[str, Any]:
    """
    Convert a Blueprint object into a Python dictionary suitable for JSON/YAML export.

    This function will always store the station function as a dotted import path (import_path).
    If the station was originally given a direct Python function (rather than an import_path),
    we attempt to reconstruct it as: "{st_cfg.func.__module__}.{st_cfg.func.__name__}".
    This is the behavior of StationConfig.to_dict

    Parameters
    ----------
    blueprint : Blueprint
        The blueprint to export.

    Returns
    -------
    dict[str, Any]
        A dictionary containing all data necessary to reconstruct the blueprint in another environment.
    """
    data: dict[str, Any] = {}
    data["entry_station"] = blueprint.entry_station
    data["global_error_station"] = blueprint.global_error_station

    data["stations"] = {}
    for st, st_cfg in blueprint.stations.items():
        station_dict = st_cfg.to_dict()
        station_dict["name"] = st
        data["stations"][st] = station_dict

    return data


def blueprint_from_dict(data: dict[str, Any]) -> Blueprint:
    """
    Reconstruct a Blueprint object from a Python dictionary.

    Assumes the environment in which this function is running has the same importable
    modules that contain the station functions specified in 'function' fields.
    We do not verify that the functions are importable, we rely on dynamic import at runtime.

    Parameters
    ----------
    data : dict[str, Any]
        Must contain:
            - "entry_station": str
            - "stations": dict[str, dict]
                and for each station dict:
                    {
                        "name": str,
                        "function": str (import path),
                        "transitions": dict[str, str],
                        "finish_on": list,
                        "on_error": str or None,
                    }

    Returns
    -------
    Blueprint
        A fresh Blueprint object with all stations.
        The station functions will be loaded dynamically via import path.
    """
    blueprint = Blueprint()

    stations_data = data.get("stations", {})
    for st, st_cfg_dict in stations_data.items():
        import_path = st_cfg_dict["function"]
        transitions = st_cfg_dict.get("transitions", {})
        finish_on = st_cfg_dict.get("finish_on", [])
        on_error = st_cfg_dict.get("on_error", None)

        blueprint.add_station(
            name=st,
            import_path=import_path,
            transitions=transitions,
            finish_on=finish_on,
            on_error=on_error,
        )

    entry_st = data.get("entry_station", None)
    if entry_st is not None:
        blueprint.set_entry_station(entry_st)
    else:
        raise ValueError(
            "No 'entry_station' found in blueprint data. Cannot restore Blueprint."
        )
    global_error_station = data.get("global_error_station", None)
    if global_error_station:
        blueprint.set_global_error_station(global_error_station)

    return blueprint


def blueprint_to_yaml(blueprint: Blueprint, filepath: str) -> None:
    """
    Serialize a Blueprint object to a YAML file.

    Parameters
    ----------
    blueprint : Blueprint
        The blueprint to export.
    filepath : str
        The file path to save the YAML.
    """
    data = blueprint_to_dict(blueprint)
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"Blueprint exported to {filepath}")


def blueprint_from_yaml(filepath: str) -> Blueprint:
    """
    Deserialize a Blueprint from a YAML file.

    Parameters
    ----------
    filepath : str
        A YAML file path that contains the serialized blueprint data.

    Returns
    -------
    Blueprint
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    blueprint = blueprint_from_dict(data)
    print(f"Blueprint imported from {filepath}")
    return blueprint


def blueprint_to_json(blueprint: Blueprint, filepath: str) -> None:
    """
    Serialize a Blueprint object to a JSON file.

    Parameters
    ----------
    blueprint : Blueprint
        The blueprint to export.
    filepath : str
        The file path to save the JSON.
    """
    data = blueprint_to_dict(blueprint)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Blueprint exported to {filepath}")


def blueprint_from_json(filepath: str) -> Blueprint:
    """
    Deserialize a Blueprint from a JSON file.

    Parameters
    ----------
    filepath : str
        A JSON file path that contains the serialized blueprint data.

    Returns
    -------
    Blueprint
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    blueprint = blueprint_from_dict(data)
    print(f"Blueprint imported from {filepath}")
    return blueprint
