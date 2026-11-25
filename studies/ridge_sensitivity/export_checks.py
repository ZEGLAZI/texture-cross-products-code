"""Export calculated, unrounded table cells for the unified PAA PDF audit."""
import argparse
import csv
import json
from pathlib import Path

NAMES = {"H": "Colour–texture", "R": "Supervised ResNet-50", "DR": "DINO ResNet-50",
         "DV": "DINO ViT-S/16", "D": "DINOv2 ViT-S/14"}


def export(source, out):
    report = json.loads(source.read_text())
    cells = []
    for item in report["checks"]:
        cell = dict(item)
        if cell["table"] == "S40" and cell["field"].startswith(("extended_", "cap_tied_")):
            cell["decimals"] = 0
        if cell["table"] == "S41" and cell["field"] == "selected_lambda":
            cell["decimals"] = 1
        cell["row_label"] = NAMES.get(cell["row"], cell["row"].replace("DTD_", "DTD "))
        cells.append(cell)
    assert len(cells) == 232
    tables = []
    for table in ("4", "S38", "S39", "S40", "S41"):
        entries = [c for c in cells if c["table"] == table]
        rows = []
        for label in dict.fromkeys(c["row"] for c in entries):
            selected = [c for c in entries if c["row"] == label]
            rows.append({"row": label, "row_label": selected[0]["row_label"],
                         "fields": [c["field"] for c in selected],
                         "values": [c["observed"] for c in selected],
                         "decimals": [c["decimals"] for c in selected]})
        tables.append({"table": table, "rows": rows})
    out.mkdir(parents=True, exist_ok=True)
    payload = {"source": str(source), "numeric_values": len(cells), "tables": tables}
    (out / "published_table_values.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False)+"\n")
    with (out / "published_table_values.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=folder / "audit_v3.json")
    parser.add_argument("--output", type=Path, default=folder)
    args = parser.parse_args()
    export(args.input, args.output)
    print(args.output / "published_table_values.json")
