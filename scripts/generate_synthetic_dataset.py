"""
Generates a synthetic TRACE-X style Bitcoin dataset for local testing/demo purposes.
This is ONLY used so the pipeline can be validated end-to-end before pointing it at
the real ~/bitcoin-monitor/data/trace_x_bitcoin_12000.json file.

Usage:
    python scripts/generate_synthetic_dataset.py --n 12000 --out data/raw/trace_x_bitcoin_12000.json
"""
import argparse
import ipaddress
import json
import random
import string
import uuid
from datetime import datetime, timedelta, timezone

COUNTRIES = ["US", "RU", "CN", "DE", "NG", "IN", "BR", "NL", "IR", "KP", "GB", "FR", "UA", "SG"]
SCRIPT_TYPES = ["P2PKH", "P2SH", "P2WPKH", "P2WSH", "P2TR"]
ANOMALY_TYPES = [
    "transaction_burst",
    "rapid_fan_out",
    "rapid_consolidation",
    "dormant_wallet_awakening",
    "shared_ip_infrastructure",
    "rapid_forwarding",
]
ANOMALY_FAMILY = {
    "transaction_burst": "velocity",
    "rapid_fan_out": "structuring",
    "rapid_consolidation": "structuring",
    "dormant_wallet_awakening": "behavioral_shift",
    "shared_ip_infrastructure": "infrastructure",
    "rapid_forwarding": "layering",
}


def rand_ip():
    return str(ipaddress.IPv4Address(random.randint(0x0A000001, 0xDF000000)))


def rand_wallet():
    return "1" + "".join(random.choices(string.ascii_letters + string.digits, k=33))


def rand_txid():
    return uuid.uuid4().hex + uuid.uuid4().hex[:32]


def gen_record(base_time, is_anomalous, shared_ip_pool, dormant_wallet_pool):
    ts = base_time + timedelta(seconds=random.randint(0, 60 * 60 * 24 * 30))
    n_in = random.randint(1, 3)
    n_out = random.randint(1, 3)

    anomaly_type = None
    anomaly_family = None

    if is_anomalous:
        anomaly_type = random.choice(ANOMALY_TYPES)
        anomaly_family = ANOMALY_FAMILY[anomaly_type]
        if anomaly_type == "rapid_fan_out":
            n_out = random.randint(8, 20)
        elif anomaly_type == "rapid_consolidation":
            n_in = random.randint(8, 20)
        elif anomaly_type == "shared_ip_infrastructure":
            src_ip = random.choice(shared_ip_pool)
        elif anomaly_type == "dormant_wallet_awakening":
            pass  # handled via wallet pool below

    src_ip = random.choice(shared_ip_pool) if (is_anomalous and anomaly_type == "shared_ip_infrastructure") else rand_ip()
    dst_ip = rand_ip()

    input_addresses = [random.choice(dormant_wallet_pool) if (is_anomalous and anomaly_type == "dormant_wallet_awakening" and i == 0) else rand_wallet() for i in range(n_in)]
    output_addresses = [rand_wallet() for _ in range(n_out)]

    base_amt = random.uniform(0.001, 2.5)
    if is_anomalous and anomaly_type == "transaction_burst":
        base_amt = random.uniform(0.0001, 0.01)

    input_amounts = [round(base_amt * random.uniform(0.5, 1.5), 8) for _ in range(n_in)]
    output_amounts = [round(base_amt * random.uniform(0.4, 1.4), 8) for _ in range(n_out)]

    fee = round(random.uniform(0.00001, 0.001), 8)

    record = {
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": random.randint(1024, 65535),
        "dst_port": 8333,
        "txid": rand_txid(),
        "input_addresses": input_addresses,
        "output_addresses": output_addresses,
        "input_amounts": input_amounts,
        "output_amounts": output_amounts,
        "fee": fee,
        "script_type": random.choice(SCRIPT_TYPES),
        "geo_country": random.choice(COUNTRIES),
        "asn": f"AS{random.randint(1000, 65000)}",
        "transaction_status": random.choices(["confirmed", "pending", "failed"], weights=[0.9, 0.08, 0.02])[0],
        "synthetic_label": 1 if is_anomalous else 0,
        "anomaly_type": anomaly_type,
        "anomaly_family": anomaly_family,
    }
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--anomaly_frac", type=float, default=0.2)
    ap.add_argument("--out", type=str, default="data/raw/trace_x_bitcoin_12000.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

    shared_ip_pool = [rand_ip() for _ in range(6)]
    dormant_wallet_pool = [rand_wallet() for _ in range(15)]

    n_anom = int(args.n * args.anomaly_frac)
    flags = [True] * n_anom + [False] * (args.n - n_anom)
    random.shuffle(flags)

    records = [gen_record(base_time, f, shared_ip_pool, dormant_wallet_pool) for f in flags]

    # inject a few messy rows to test cleaning (missing/invalid values, duplicates)
    if len(records) > 20:
        records[5]["src_ip"] = "999.999.999.999"  # invalid IP
        records[6]["timestamp"] = "not-a-date"     # invalid timestamp
        records[7]["fee"] = None                    # missing fee
        records[8] = dict(records[7])                # duplicate-ish row
        records[9]["input_addresses"] = []            # empty inputs

    with open(args.out, "w") as f:
        json.dump(records, f, indent=1)

    print(f"Wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
