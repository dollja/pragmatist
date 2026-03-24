from __future__ import annotations

from pathlib import Path
import argparse
import json

from advanced_balanced_hybrid_persona_pipeline import demo_run as advanced_demo_run
from pragmatist_hybrid_persona_integration import demo_integration


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete Hybrid Persona + Pragmatist demo package."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/generated_demo_run",
        help="Directory where the demo artifacts will be written.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    codebook_path = root / "hybrid_persona_label_codebook.csv"
    schema_path = root / "hybrid_persona_schema.json"

    advanced_dir = output_dir / "advanced_balanced_pipeline"
    pragmatist_dir = output_dir / "pragmatist_runtime"

    advanced_result = advanced_demo_run(
        codebook_path=codebook_path,
        schema_path=schema_path,
        output_dir=advanced_dir,
    )

    pragmatist_result = demo_integration(
        codebook_path=codebook_path,
        schema_path=schema_path,
        output_dir=pragmatist_dir,
    )

    summary = {
        "package_root": str(root),
        "output_dir": str(output_dir),
        "advanced_balanced_pipeline": {
            "paths": advanced_result["paths"],
        },
        "pragmatist_runtime": pragmatist_result,
    }

    summary_path = output_dir / "complete_pipeline_demo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\nComplete demo finished.")
    print("Summary:", summary_path)
    print("Advanced outputs:", advanced_dir)
    print("Pragmatist outputs:", pragmatist_dir)


if __name__ == "__main__":
    main()
