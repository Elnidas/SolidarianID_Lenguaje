#!/usr/bin/env python3
"""
maria_metrics_report.py
Revisa una carpeta y genera para cada *.json*:
  • el informe en texto alineado (igual que el .txt original) y
  • un bloque ```markdown``` listo para copiar/pegar.

Uso:
    # 1) En la misma carpeta que los .json
    python maria_metrics_report.py

    # 2) Indicando carpeta
    python maria_metrics_report.py /ruta/a/carpeta

La salida se imprime en stdout, separada por cabeceras de fichero.
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

DECIMALS = 4  # número de decimales en valores float


# ---------- Helpers ---------------------------------------------------------
def _fmt_line(lbl: str, row: Dict[str, Any], d: int = DECIMALS) -> str:
    return (
        f"{lbl:>11}"
        f" {row['precision']:>10.{d}f}"
        f" {row['recall']:>10.{d}f}"
        f" {row['f1-score']:>10.{d}f}"
        f" {int(row['support']):>10}"
    )


def build_md(m: Dict[str, Any]) -> str:
    cr, acc = m["classification_report"], m["accuracy"]
    cm = m["confusion_matrix"]
    inf = m.get("inference", {})

    md = [
        "| label | precision | recall | f1-score | support |",
        "|-------|-----------|--------|----------|---------|",
        f"| 0 | {cr['0']['precision']:.{DECIMALS}f} | {cr['0']['recall']:.{DECIMALS}f} | {cr['0']['f1-score']:.{DECIMALS}f} | {int(cr['0']['support'])} |",
        f"| 1 | {cr['1']['precision']:.{DECIMALS}f} | {cr['1']['recall']:.{DECIMALS}f} | {cr['1']['f1-score']:.{DECIMALS}f} | {int(cr['1']['support'])} |",
        f"| accuracy |  |  | {acc:.{DECIMALS}f} | {int(cr['macro avg']['support'])} |",
        f"| macro avg | {cr['macro avg']['precision']:.{DECIMALS}f} | {cr['macro avg']['recall']:.{DECIMALS}f} | {cr['macro avg']['f1-score']:.{DECIMALS}f} | {int(cr['macro avg']['support'])} |",
        f"| weighted avg | {cr['weighted avg']['precision']:.{DECIMALS}f} | {cr['weighted avg']['recall']:.{DECIMALS}f} | {cr['weighted avg']['f1-score']:.{DECIMALS}f} | {int(cr['weighted avg']['support'])} |",
        "",
        "| | pred 0 | pred 1 |",
        "|---|-------|-------|",
        f"| real 0 | {cm['pred 0']['real 0']} | {cm['pred 1']['real 0']} |",
        f"| real 1 | {cm['pred 0']['real 1']} | {cm['pred 1']['real 1']} |",
    ]

    if inf:
        md.extend(
            [
                "",
                f"**Tiempo inferencia:** {inf['total_seconds']:.2f} s "
                f"({inf['per_sample_seconds']:.{DECIMALS}f} s/ej.)  ",
                f"**Pico CPU:** {inf['peak_cpu_mb']:.1f} MB  "
                f"(RSS final {inf['rss_after_mb']:.1f} MB)  ",
            ]
        )
        if inf.get("peak_gpu_mb"):
            md.append(f"**Pico GPU:** {inf['peak_gpu_mb']:.1f} MB  ")

    return "\n".join(md)


def report_file(fp: Path) -> None:
    with fp.open(encoding="utf-8") as f:
        metrics = json.load(f)

    print("=" * 80)
    print(f"== {fp.name} ==")
    print("=" * 80)

    print(build_md(metrics))



# ---------- Main ------------------------------------------------------------
def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    if not folder.is_dir():
        sys.exit(f"{folder} no es un directorio válido")

    json_files: List[Path] = sorted(folder.glob("*.json"))
    if not json_files:
        sys.exit(f"No se encontraron .json en {folder}")

    for fp in json_files:
        report_file(fp)


if __name__ == "__main__":
    main()
