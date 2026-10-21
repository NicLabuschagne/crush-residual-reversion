"""
Fetch the exact contract behind each continuous series (free Databento endpoint).

Concept: a continuous symbol like ZS.v.0 is a pointer that switches between
real contracts (ZSN1, ZSX1, ...) over time. symbology.resolve returns those
switch dates, so we know exactly which contract every bar came from and where
each roll gap sits. The call is free; it does not use data credits.

Databento does not map continuous -> contract code directly, so this runs in
two steps:
  1. continuous symbol -> instrument_id   (with the date range of each segment)
  2. instrument_id     -> raw_symbol      (the contract code, e.g. ZSN1)

We resolve three roll rules (volume, open interest, calendar) because we do
not know for sure which one the original file was pulled with.

Run:  python get_roll_map.py
Then send back roll_map.json.
"""
import json
import os
import databento as db

API_KEY = os.environ["DATABENTO_API_KEY"]
START_DATE = "2020-06-25"
END_DATE = "2025-12-17"

client = db.Historical(API_KEY)

legs = ["ZS", "ZM", "ZL"]
roll_rules = ["v", "n", "c"]  # v = volume, n = open interest, c = calendar
continuous_symbols = [f"{leg}.{rule}.0" for leg in legs for rule in roll_rules]

# Step 1: which instrument each continuous symbol pointed to, and when
step_one = client.symbology.resolve(
    dataset="GLBX.MDP3",
    symbols=continuous_symbols,
    stype_in="continuous",
    stype_out="instrument_id",
    start_date=START_DATE,
    end_date=END_DATE,
)

# Collect every distinct instrument id that appeared in any segment
instrument_ids = sorted({
    segment["s"]
    for segments in step_one["result"].values()
    for segment in segments
})

# Step 2: translate instrument ids to contract codes
step_two = client.symbology.resolve(
    dataset="GLBX.MDP3",
    symbols=instrument_ids,
    stype_in="instrument_id",
    stype_out="raw_symbol",
    start_date=START_DATE,
    end_date=END_DATE,
)

# Build a simple id -> contract code lookup (one code per id on CME)
id_to_contract = {
    instrument_id: segments[0]["s"]
    for instrument_id, segments in step_two["result"].items()
    if segments
}

# Attach the contract code to every continuous segment
roll_map = {}
for symbol, segments in step_one["result"].items():
    roll_map[symbol] = [
        {
            "start": segment["d0"],
            "end": segment["d1"],
            "instrument_id": segment["s"],
            "contract": id_to_contract.get(segment["s"], "UNKNOWN"),
        }
        for segment in segments
    ]

with open("roll_map.json", "w") as file_handle:
    json.dump(roll_map, file_handle, indent=2)

# Quick summary so you can see it worked
for symbol, segments in roll_map.items():
    contracts = [segment["contract"] for segment in segments]
    print(f"{symbol}: {len(segments)} segments, first {contracts[:3]}")
print("step 1:", step_one["message"], "| step 2:", step_two["message"])