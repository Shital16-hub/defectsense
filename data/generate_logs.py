"""
Generate 500 realistic synthetic maintenance logs for DefectSense RAG corpus.

Run:
    python data/generate_logs.py
"""
import csv
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent
OUTPUT_PATH = DATA_DIR / "maintenance_logs.csv"

random.seed(42)

# ── Domain Data ───────────────────────────────────────────────────────────────

FAILURE_TYPES = {
    "TWF": {
        "label": "Tool Wear Failure",
        "symptoms": [
            "Tool wear exceeded 200 minutes, surface finish degradation observed",
            "Cutting tool showing micro-fractures, dimensional tolerance drifting",
            "Excessive vibration during machining, tool edge chipping detected",
            "Increased spindle load above 85%, tool wear at 185+ minutes",
            "Workpiece surface roughness above specification, tool flank wear",
        ],
        "root_causes": [
            "Tool reached end of rated service life without scheduled replacement",
            "Incorrect cutting parameters (feed rate too high) accelerated wear",
            "Abrasive workpiece material caused premature tool degradation",
            "Insufficient coolant flow to cutting zone increased thermal wear",
            "Vibration from worn spindle bearings caused uneven tool wear pattern",
        ],
        "actions": [
            "Replaced cutting insert, calibrated spindle to specification",
            "Installed new carbide tool, adjusted feed rate per manufacturer spec",
            "Replaced tool assembly, reviewed cutting parameter sheet with operator",
            "Swapped worn tool for fresh insert, flushed coolant lines",
            "Replaced tool and spindle bearing, updated maintenance schedule",
        ],
    },
    "HDF": {
        "label": "Heat Dissipation Failure",
        "symptoms": [
            "Process temperature exceeded 315K, cooling fan audibly slower",
            "Temperature differential between air and process exceeded 12K",
            "Thermal alarm triggered, machine auto-shutdown at 320K",
            "Process temp rising 2K above normal, coolant return temp elevated",
            "Overheating warning on HMI, fan speed reading below threshold",
        ],
        "root_causes": [
            "Cooling fan blade fractured, reducing airflow by 60%",
            "Heat exchanger fins fouled with metal shavings, impeding cooling",
            "Coolant pump seal failure causing loss of coolant pressure",
            "Blocked air intake filter restricting cooling airflow",
            "Cooling system thermostat failed closed, bypassing temperature regulation",
        ],
        "actions": [
            "Replaced cooling fan assembly, cleaned heat exchange fins",
            "Cleaned and descaled heat exchanger, replaced coolant filter",
            "Replaced coolant pump seal, pressure tested system",
            "Replaced air intake filter, cleared debris from cooling vents",
            "Replaced thermostat, flushed coolant circuit, verified temperature control",
        ],
    },
    "PWF": {
        "label": "Power Failure",
        "symptoms": [
            "Power consumption outside 3500–9000 W operating range",
            "Machine tripped breaker, power draw spiked to 11 kW briefly",
            "Rotational speed dropped 400 RPM with simultaneous torque spike",
            "Intermittent power fluctuation, VFD showing under-voltage fault",
            "Motor current draw 40% above rated, thermal overload tripped",
        ],
        "root_causes": [
            "Variable frequency drive (VFD) capacitor bank degradation",
            "Phase imbalance in power supply causing motor overload",
            "Motor winding insulation breakdown from thermal cycling",
            "Loose power cable connection causing resistance heating",
            "Power factor correction capacitor failure, increasing reactive load",
        ],
        "actions": [
            "Replaced VFD capacitor bank, re-calibrated drive parameters",
            "Balanced three-phase supply, inspected and tightened all connections",
            "Rewound motor armature, applied thermal barrier coating",
            "Torqued all power cable terminations, applied dielectric grease",
            "Replaced PFC capacitor bank, verified power quality with analyser",
        ],
    },
    "OSF": {
        "label": "Overstrain Failure",
        "symptoms": [
            "Torque exceeded 72 Nm at low RPM, machine jerk detected",
            "Gearbox making grinding noise, torque sensor showing oscillations",
            "Spindle locked momentarily, torque spike to 80 Nm recorded",
            "Mechanical shudder during high-load operation, torque instability",
            "Tool holder showing signs of fretting, elevated torque readings",
        ],
        "root_causes": [
            "Work material hardness exceeded machine rated torque capacity",
            "Incorrect gear mesh causing intermittent binding under load",
            "Worn drive belt slipping under high torque causing intermittent overload",
            "Spindle bearing pre-load out of specification, binding at high torque",
            "Machining parameters set beyond material/tool combination limits",
        ],
        "actions": [
            "Adjusted cutting depth and feed rate, verified torque limits",
            "Re-shimmed gearbox, replaced worn gear set, re-greased bearings",
            "Replaced drive belt and tensioner, verified torque transmission",
            "Adjusted spindle bearing pre-load to specification, re-greased",
            "Revised CNC program to comply with tool/material torque limits",
        ],
    },
    "RNF": {
        "label": "Random Failure",
        "symptoms": [
            "Unexpected machine stop, no sensor pre-warning observed",
            "Random fault code on PLC, no reproducible trigger found",
            "Intermittent E-stop activation, cause unclear after investigation",
            "Machine halted mid-cycle, all sensors within normal range at time",
            "Unscheduled downtime, no clear sensor anomaly precursor identified",
        ],
        "root_causes": [
            "Loose encoder cable caused spurious position fault",
            "Electromagnetic interference triggered false safety circuit trip",
            "PLC memory bit corruption caused unexpected program branch",
            "Intermittent contactor contact bounce triggered motor protection",
            "Software watchdog timeout due to network latency spike",
        ],
        "actions": [
            "Secured encoder cable, wrapped with shielded conduit",
            "Installed EMI filter on control cabinet power input",
            "Performed PLC memory checksum verification, cleared fault log",
            "Replaced contactor, tested safety circuit function",
            "Updated PLC firmware, increased watchdog timeout parameter",
        ],
    },
}

MACHINE_IDS = [f"M{i:03d}" for i in range(1, 51)]  # M001–M050
MACHINE_TYPES = ["L", "M", "H"]
TECHNICIANS = [
    "J. Smith", "A. Kumar", "L. Chen", "R. Patel", "M. Johansson",
    "T. Williams", "S. Nakamura", "B. Mueller", "C. Santos", "D. Okafor",
]

RESOLUTION_TIME_BY_FAILURE = {
    "TWF": (0.5, 4.0),   # quick — just replace insert
    "HDF": (2.0, 8.0),   # moderate — cooling system work
    "PWF": (3.0, 12.0),  # longer — electrical diagnosis
    "OSF": (1.0, 6.0),   # moderate — mechanical adjustment
    "RNF": (0.5, 3.0),   # short — often reset/clean
}

START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2025, 1, 1)


def random_date() -> datetime:
    delta = END_DATE - START_DATE
    return START_DATE + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def generate_log(index: int) -> dict:
    machine_id = random.choice(MACHINE_IDS)
    failure_type = random.choices(
        list(FAILURE_TYPES.keys()),
        weights=[25, 20, 15, 20, 20],  # roughly realistic proportions
        k=1,
    )[0]
    fdata = FAILURE_TYPES[failure_type]
    min_hours, max_hours = RESOLUTION_TIME_BY_FAILURE[failure_type]

    return {
        "log_id": str(uuid.uuid4()),
        "machine_id": machine_id,
        "machine_type": random.choice(MACHINE_TYPES),
        "date": random_date().isoformat(),
        "failure_type": failure_type,
        "failure_type_label": fdata["label"],
        "symptoms": random.choice(fdata["symptoms"]),
        "root_cause": random.choice(fdata["root_causes"]),
        "action_taken": random.choice(fdata["actions"]),
        "resolution_time_hours": round(random.uniform(min_hours, max_hours), 2),
        "technician": random.choice(TECHNICIANS),
    }


def main() -> None:
    print("Generating 500 synthetic maintenance logs...")
    logs = [generate_log(i) for i in range(500)]

    fieldnames = list(logs[0].keys())
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(logs)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 55)
    print("  Synthetic Maintenance Logs — Summary")
    print("=" * 55)
    print(f"  Total records : {len(logs)}")
    print(f"  Saved to      : {OUTPUT_PATH}")
    print()
    from collections import Counter
    type_counts = Counter(log["failure_type"] for log in logs)
    print("  Failure Type Distribution:")
    for ft, count in sorted(type_counts.items()):
        label = FAILURE_TYPES[ft]["label"]
        bar = "█" * (count // 5)
        print(f"    {ft} ({label:<26}): {count:3d}  {bar}")

    print()
    machine_counts = Counter(log["machine_id"] for log in logs)
    top5 = machine_counts.most_common(5)
    print("  Top 5 Machines by Incident Count:")
    for mid, count in top5:
        print(f"    {mid}: {count} incidents")
    print("=" * 55)


if __name__ == "__main__":
    main()
