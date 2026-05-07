"""
Generate 500 realistic synthetic maintenance logs for DefectSense RAG corpus.
Covers the four Azure PdM failure types: EVF, RSF, PBF, BWF.

Run:
    python data/generate_logs.py
"""
import csv
import random
import uuid
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR    = Path(__file__).parent
OUTPUT_PATH = DATA_DIR / "maintenance_logs.csv"

random.seed(42)

# ── Domain Data ───────────────────────────────────────────────────────────────

FAILURE_TYPES = {
    "EVF": {
        "label": "Electrical Overload Failure",
        "symptoms": [
            "Voltage spike detected, volt reading 15 to 25 units above normal baseline",
            "Electrical overload alarm triggered, volt sensor exceeded safe operating range",
            "Abnormal power surge recorded, voltage fluctuation above rated capacity",
            "Control panel showed overvoltage warning, volt reading unstable",
            "Electrical fault detected, voltage consistently above normal threshold",
        ],
        "root_causes": [
            "Power supply unit degradation causing voltage regulation failure",
            "Loose electrical connection in main power circuit causing intermittent surges",
            "Capacitor bank failure in voltage regulator allowing uncontrolled spikes",
            "External grid voltage fluctuation exceeding equipment tolerance",
            "Motor winding insulation breakdown causing electrical leakage",
        ],
        "actions": [
            "Replaced power supply unit and tested voltage regulation",
            "Tightened all electrical connections and applied dielectric grease",
            "Replaced capacitor bank in voltage regulator circuit",
            "Installed voltage surge protector on main power input",
            "Rewound motor armature and applied new insulation coating",
        ],
    },
    "RSF": {
        "label": "Rotor Speed Failure",
        "symptoms": [
            "Rotation speed dropped 60 to 90 RPM below target, motor struggling to maintain speed",
            "Rotor slowdown detected, rotate sensor showing persistent speed loss",
            "Motor unable to maintain rated RPM under load, speed gradually declining",
            "Abnormal speed fluctuation recorded, rotate reading unstable under normal load",
            "Speed loss alarm triggered, rotor rotation below minimum threshold",
        ],
        "root_causes": [
            "Drive belt worn and slipping causing rotational power loss",
            "Motor bearing seizure increasing mechanical resistance to rotation",
            "Variable frequency drive fault causing incorrect speed command",
            "Rotor winding degradation reducing motor torque output",
            "Mechanical coupling looseness causing intermittent power transmission loss",
        ],
        "actions": [
            "Replaced drive belt and tensioner assembly",
            "Replaced seized motor bearing and regreased assembly",
            "Reprogrammed variable frequency drive parameters",
            "Rewound rotor winding and tested motor output torque",
            "Tightened mechanical coupling and verified alignment",
        ],
    },
    "PBF": {
        "label": "Pressure Blockage Failure",
        "symptoms": [
            "Pressure reading 20 to 30 units above normal, possible line blockage",
            "Pressure spike alarm triggered, flow restricted in main circuit",
            "Abnormal pressure buildup detected, pressure sensor exceeding safe limit",
            "Blockage warning activated, pressure rising progressively over several hours",
            "System pressure unstable, pressure sensor showing erratic high readings",
        ],
        "root_causes": [
            "Foreign debris blocking main pressure line causing flow restriction",
            "Valve seizure in closed position preventing normal pressure relief",
            "Filter element completely clogged restricting fluid flow",
            "Pump outlet check valve stuck closed causing pressure buildup",
            "Pipe corrosion buildup reducing internal diameter causing restriction",
        ],
        "actions": [
            "Flushed main pressure line and removed foreign debris",
            "Replaced seized valve assembly and tested pressure relief",
            "Replaced clogged filter element and flushed circuit",
            "Replaced stuck check valve and verified pump outlet flow",
            "Descaled and cleaned pipe section and replaced corroded segment",
        ],
    },
    "BWF": {
        "label": "Bearing Wear Failure",
        "symptoms": [
            "Vibration level 8 to 12 units above normal baseline, bearing wear suspected",
            "Abnormal vibration detected, vibration sensor showing gradual increase over days",
            "Bearing noise reported by operator, vibration reading consistently elevated",
            "Mechanical vibration alarm triggered, vibration exceeding acceptable threshold",
            "Progressive vibration increase recorded, bearing degradation pattern confirmed",
        ],
        "root_causes": [
            "Rolling element bearing fatigue causing increased vibration signature",
            "Insufficient lubrication causing bearing surface wear and vibration",
            "Bearing misalignment causing uneven load distribution and vibration",
            "Contamination of bearing lubricant causing accelerated wear",
            "Bearing overloaded beyond rated capacity causing premature fatigue",
        ],
        "actions": [
            "Replaced worn bearing assembly and verified alignment",
            "Cleaned and relubricated bearing assembly with fresh grease",
            "Realigned bearing assembly and verified with dial indicator",
            "Flushed contaminated lubricant and replaced with correct grade",
            "Replaced overloaded bearing with higher capacity rated assembly",
        ],
    },
}

# Azure PdM: 100 machines, models 1–4
MACHINE_IDS   = [f"M{i:03d}" for i in range(1, 101)]
MACHINE_TYPES = ["model1", "model2", "model3", "model4"]

TECHNICIANS = [
    "J. Smith", "A. Kumar", "L. Chen", "R. Patel", "M. Johansson",
    "T. Williams", "S. Nakamura", "B. Mueller", "C. Santos", "D. Okafor",
]

# EVF 20%, RSF 25%, PBF 30%, BWF 25%
FAILURE_WEIGHTS = {"EVF": 20, "RSF": 25, "PBF": 30, "BWF": 25}

RESOLUTION_TIME_BY_FAILURE = {
    "EVF": (2.0, 8.0),   # electrical work takes time
    "RSF": (1.0, 5.0),   # mechanical repair moderate
    "PBF": (0.5, 3.0),   # clearing blockage usually quick
    "BWF": (2.0, 6.0),   # bearing replacement takes time
}

START_DATE = datetime(2015, 1, 1)
END_DATE   = datetime(2016, 1, 1)


def random_date() -> datetime:
    delta = END_DATE - START_DATE
    return START_DATE + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def generate_log(index: int) -> dict:
    failure_type = random.choices(
        list(FAILURE_WEIGHTS.keys()),
        weights=list(FAILURE_WEIGHTS.values()),
        k=1,
    )[0]
    fdata = FAILURE_TYPES[failure_type]
    min_hours, max_hours = RESOLUTION_TIME_BY_FAILURE[failure_type]

    return {
        "log_id":                 str(uuid.uuid4()),
        "machine_id":             random.choice(MACHINE_IDS),
        "machine_type":           random.choice(MACHINE_TYPES),
        "date":                   random_date().isoformat(),
        "failure_type":           failure_type,
        "failure_type_label":     fdata["label"],
        "symptoms":               random.choice(fdata["symptoms"]),
        "root_cause":             random.choice(fdata["root_causes"]),
        "action_taken":           random.choice(fdata["actions"]),
        "resolution_time_hours":  round(random.uniform(min_hours, max_hours), 2),
        "technician":             random.choice(TECHNICIANS),
    }


def main() -> None:
    print("Generating 500 synthetic maintenance logs (Azure PdM failure types)...")
    logs = [generate_log(i) for i in range(500)]

    fieldnames = list(logs[0].keys())
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(logs)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Synthetic Maintenance Logs — Summary")
    print("=" * 60)
    print(f"  Total records : {len(logs)}")
    print(f"  Saved to      : {OUTPUT_PATH}")
    print()

    type_counts = Counter(log["failure_type"] for log in logs)
    print("  Failure Type Distribution:")
    for ft in ["EVF", "RSF", "PBF", "BWF"]:
        count = type_counts[ft]
        label = FAILURE_TYPES[ft]["label"]
        bar   = "#" * (count // 5)
        print(f"    {ft} ({label:<30}): {count:3d}  {bar}")

    print()
    machine_counts = Counter(log["machine_id"] for log in logs)
    top5 = machine_counts.most_common(5)
    print("  Top 5 Machines by Incident Count:")
    for mid, count in top5:
        print(f"    {mid}: {count} incidents")
    print("=" * 60)


if __name__ == "__main__":
    main()
