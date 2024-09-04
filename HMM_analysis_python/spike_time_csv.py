# Spike Time CSVs
#
# A script which takes an experiment ID and a canonical name, loads the data from S3,
# and produces two CSV files: $name.csv with the spike times, and $name.backbone.csv
# which just marks whether each unit is part of the backbone or not.
import argparse
import os
import sys

import pandas as pd

from hmmsupport import get_raster, load_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="spike_time_csvs",
        description="Convert data to CSV for criticality pipeline",
    )
    parser.add_argument("name")
    parser.add_argument("exp")
    parser.add_argument("--force", "-f", action="store_true")

    args = parser.parse_args()
    csvname = args.name + ".csv"
    bbname = args.name + "_backbone.csv"

    if not args.force and os.path.isfile(csvname) and os.path.isfile(bbname):
        print(f"Skipping {args.exp} because output CSV for {args.name} exists.")
        sys.exit(0)

    if not os.environ.get("S3_USER"):
        print("$S3_USER must be defined.", file=sys.stderr)
        sys.exit(1)

    print(f"Writing {args.exp} to {csvname} and {bbname}")

    # Load the raster (bin size doesn't affect the output CSV) and generate the CSV
    # where each row is a single spike.
    r = get_raster("org_and_slice", args.exp, 30)
    pd.DataFrame(
        dict(unit=i, time=t) for i, ts in enumerate(r.train) for t in ts
    ).to_csv(csvname, index=False)

    # Load the metrics and generate a CSV out of the metrics as well.
    metrics = load_metrics(args.exp, only_include=["scaf_units"])
    scaf = set(metrics["scaf_units"].ravel())
    pd.DataFrame(dict(unit=i, backbone=i + 1 in scaf) for i in range(r.N)).to_csv(
        bbname, index=False
    )
