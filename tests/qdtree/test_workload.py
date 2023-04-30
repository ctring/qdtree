import pandas as pd
import pytest
from qdtree import Workload


@pytest.fixture
def workload():
    workload_dict = {
        "schema": {
            "x": "int",
            "y": "float",
            "z": "date",
        },
        "queries": {
            "q1": {
                "type": "expr",
                "children": ["z", ">=", "2020-01-01"],
            },
            "q2": {
                "type": "and",
                "children": [
                    {
                        "type": "expr",
                        "children": ["x", ">", "10"],
                    },
                    {
                        "type": "expr",
                        "children": ["y", "<", "1.5"],
                    },
                ],
            },
            "q3": {
                "type": "or",
                "children": [
                    {
                        "type": "and",
                        "children": [
                            {
                                "type": "expr",
                                "children": ["x", ">", "10"],
                            },
                            {
                                "type": "expr",
                                "children": ["y", ">=", "0.5"],
                            },
                        ],
                    },
                    {
                        "type": "expr",
                        "children": ["z", "<=", "2020-02-01"],
                    },
                ],
            },
        },
    }
    return Workload(workload_dict)


def test_print_workload(workload):
    print(workload)


def test_workload_cut_repo(workload):
    repo = workload.cut_repo

    assert len(repo) == 5

    assert str(repo.get("z", ">=", "2020-01-01")) == "z >= 2020-01-01 00:00:00"
    assert str(repo.get("x", ">", "10")) == "x > 10"
    assert str(repo.get("y", "<", "1.5")) == "y < 1.5"
    assert str(repo.get("y", ">=", "0.5")) == "y >= 0.5"
    assert str(repo.get("z", "<=", "2020-02-01")) == "z <= 2020-02-01 00:00:00"


def test_workload_queries(workload):
    queries = workload._queries

    assert len(queries) == 3

    assert str(queries["q1"]) == "z >= 2020-01-01 00:00:00"
    assert str(queries["q2"]) == "(x > 10) and (y < 1.5)"
    assert (
        str(queries["q3"]) == "((x > 10) and (y >= 0.5)) or (z <= 2020-02-01 00:00:00)"
    )
