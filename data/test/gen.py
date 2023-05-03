import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, default=1000,
                    help="Number of rows to generate")
parser.add_argument("--out", type=str, default="test.csv", help="Output file")
args = parser.parse_args()

df = pd.DataFrame(
    {
        "cpu": np.random.randint(0, 100, args.rows),
        "disk": np.random.rand(args.rows),
    }
)
df.to_csv(args.out, index=False)

print(f"Generated {args.rows} rows in {args.out}")
