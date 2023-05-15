# Example usage:
#   python3 gen_workload.py tpc-h/workload_template.json --queries tpc-h/queries --out tpc-h/workload.json
#
import argparse
import datetime
import copy
import json
import os
import random
import re
from typing import Optional

from dateutil.relativedelta import relativedelta


def process_value(val: str, match: Optional[re.Match]):
    # Replace $rand_int(min, max) with a random integer in [min, max]
    val = re.sub(
        r"\$rand_int\(([0-9]+), ([0-9]+)\)",
        lambda m: str(random.randint(int(m.group(1)), int(m.group(2)))),
        val,
    )

    # Replace $rand_float(min, max) with a random float in [min, max]
    val = re.sub(
        r"\$rand_float\(([0-9]+), ([0-9]+)\)",
        lambda m: f"{random.uniform(int(m.group(1)), int(m.group(2))):.2f}",
        val,
    )

    # Replace $rand_date(min, max) with a random date in [min, max]
    val = re.sub(
        r"\$rand_date\('([0-9-]+)', '([0-9-]+)'\)",
        lambda m: str(
            datetime.datetime.fromtimestamp(
                random.randint(
                    int(datetime.datetime.strptime(m.group(1), "%Y-%m-%d").timestamp()),
                    int(datetime.datetime.strptime(m.group(2), "%Y-%m-%d").timestamp()),
                )
            ).strftime("%Y-%m-%d")
        ),
        val,
    )

    if match:
        # Replace $1, $2, ... with the corresponding match group
        val = re.sub(r"\$([0-9]+)", lambda m: match.group(int(m.group(1))), val)

    # Compute the date expression if any
    date_expr_match = re.search(
        r"date '([0-9-]+)' ([+-]) interval '([0-9]+)' (\w+)", val
    )
    if date_expr_match:
        date, op, delta, unit = date_expr_match.groups()
        DATE_FORMAT = "%Y-%m-%d"

        def doop(val, op, delta):
            if op == "+":
                return val + delta
            elif op == "-":
                return val - delta
            else:
                raise ValueError(f"Unknown operator: {op}")

        date = datetime.datetime.strptime(date, DATE_FORMAT)
        date = doop(date, op, relativedelta(**{unit + "s": int(delta)}))
        val = f"{date.strftime(DATE_FORMAT)}"

    # Compute the arithmetic expression if any
    arith_expr_match = re.search(r"([0-9.]+) ([+-/*]) ([0-9.]+)", val)
    if arith_expr_match:
        left, op, right = arith_expr_match.groups()
        left = float(left)
        right = float(right)
        if op == "+":
            res = left + right
        elif op == "-":
            res = left - right
        elif op == "*":
            res = left * right
        elif op == "/":
            res = left / right
        else:
            raise ValueError(f"Unknown operator: {op}")
        val = f"{res:.2f}"

    return val


def substitute_predicate(node, match: Optional[re.Match]):
    node_type = node["type"]
    children = node["children"]
    if node_type in ["and", "or"]:
        for child in children:
            substitute_predicate(child, match)
    elif node_type == "expr":
        node["children"] = [process_value(child, match) for child in children]
    else:
        raise ValueError(f"Unknown node type: {node_type}")


parser = argparse.ArgumentParser()
parser.add_argument("template", help="the json file containing the workload template")
parser.add_argument("--queries", help="directory containing the queries")
parser.add_argument(
    "-n", "--num-queries", type=int, help="number of queries to generate per template"
)
parser.add_argument(
    "-o", "--output", default="workload.json", help="path to the output file"
)
args = parser.parse_args()

with open(args.template, "r") as f:
    template = json.load(f)

workload = {
    "schema": template["schema"],
    "queries": {},
}

if args.queries and args.num_queries:
    raise ValueError("Cannot specify both --queries and --num-queries")

if args.queries:
    queries = os.listdir(args.queries)

    # Iterate over the query template and extract the predicates
    for k, value in template["queries"].items():
        target_queries = [q for q in queries if q.startswith(f"{k}.")]
        regex = value["regex"]
        for q in target_queries:
            # Read and clean the query
            with open(os.path.join(args.queries, q), "r") as f:
                query = " ".join(f.read().split())
            match = re.search(regex, query)
            if match:
                new_predicate = copy.deepcopy(value["predicate"])
                substitute_predicate(new_predicate, match)
                workload["queries"][q] = new_predicate
else:
    for k, value in template["queries"].items():
        for i in range(args.num_queries):
            new_predicate = copy.deepcopy(value["predicate"])
            substitute_predicate(new_predicate, None)
            workload["queries"][f"{k}.{i}"] = new_predicate


with open(args.output, "w") as f:
    json.dump(workload, f, indent=2)
