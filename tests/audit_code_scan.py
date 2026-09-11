import re
from pathlib import Path

files = sorted(Path("dashboard/views").glob("*.py"))
print("=== Scanning View Files for Hardcoded Metrics ===")

for f in files:
    content = f.read_text(encoding="utf-8")
    metric_lines = [line.strip() for line in content.splitlines() if "st.metric(" in line]
    print(f"\nFile: {f.name} ({len(metric_lines)} metric calls)")
    for ml in metric_lines:
        print(f"  -> {ml}")
