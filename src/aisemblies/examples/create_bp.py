from aisemblies.blueprint import Blueprint
from aisemblies.examples.crag import (
    error_handler,
    generate,
    grade_documents,
    retrieve,
    transform_query,
    web_search,
)
from aisemblies.serialization import blueprint_to_yaml


def main():
    crag_blueprint = Blueprint()

    crag_blueprint.add_station(
        name="retrieve",
        func=retrieve,
        transitions={"OK": "grade_documents", "NO_QUESTION": None},
        on_error="error_handler",
    )

    crag_blueprint.add_station(
        name="grade_documents",
        func=grade_documents,
        transitions={"RELEVANT": "generate", "IRRELEVANT": "transform_query"},
        on_error="error_handler",
    )

    crag_blueprint.add_station(
        name="transform_query",
        func=transform_query,
        transitions={"TRANSFORMED": "web_search"},
        on_error="error_handler",
    )

    crag_blueprint.add_station(
        name="web_search",
        func=web_search,
        transitions={"DONE": "generate"},
        on_error="error_handler",
    )

    crag_blueprint.add_station(
        name="generate",
        func=generate,
        finish_on=["DONE"],
        on_error="error_handler",
    )

    crag_blueprint.add_station(
        name="error_handler",
        func=error_handler,
        finish_on=[None],
    )

    crag_blueprint.set_entry_station("retrieve")

    yaml_file = "src/aisemblies/examples/crag_blueprint.yaml"
    blueprint_to_yaml(crag_blueprint, yaml_file)


if __name__ == "__main__":
    main()
