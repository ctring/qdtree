import argparse
import datetime
import copy
import json
import os
import re

from dateutil.relativedelta import relativedelta


def process_value(val, match):
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
        val = f"date '{date.strftime(DATE_FORMAT)}'"

    return val


def substitute_predicate(node, match):
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
parser.add_argument("queries", help="directory containing the queries")
parser.add_argument(
    "-o", "--output", default="workload.json", help="path to the output file"
)
args = parser.parse_args()

with open(args.template, "r") as f:
    template = json.load(f)

workload = {}
queries = os.listdir(args.queries)
for k, value in template.items():
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
            workload[q] = new_predicate

with open(args.output, "w") as f:
    json.dump(workload, f, indent=2)
