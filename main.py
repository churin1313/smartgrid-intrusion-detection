"""
=============================================================================
main.py  –  Smart Grid Master Pipeline (Module 3, Core CSE – Varun)
=============================================================================
Project : Game-Theoretic Multi-Model Cyber-Physical Attack Detection for
          Smart Grid  (Dr. Priya V, SCORE, VIT)

This file is the single entry-point for the entire system.  It replaces
the old run_pipeline.py and supports two execution modes:

    CLI mode (default)
    ──────────────────
    Runs M1 → M2 in sequence, prints a full summary, and writes all outputs.

        python main.py
        python main.py --epochs 30 --temporal transformer
        python main.py --attack FDI --epochs 10 --export

    Server mode
    ───────────
    Launches the FastAPI REST server so pipelines can be triggered over HTTP.

        python main.py --mode server
        python main.py --mode server --port 8080

Full integration flow (all 5 modules):

    [M1 – Pranav]   Smart Grid signal simulation + attack injection
         ↓
    [M2 – Shashwat] Spatio-temporal GNN/LSTM detection → anomaly scores
         ↓
    [M3 – Varun]    THIS FILE: orchestration, logging, API, CSV hand-off
         ↓  ai_anomaly_scores.csv
    [M4 – Saanvi]   Multi-modal fusion + risk analytics
         ↓
    [M5 – Nithila]  Network IDS + game-theoretic defense
         ↓
    Final Output: Attack type + Severity + Optimal Defense Action
=============================================================================
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from datetime import datetime

# ── Module 3 integration core ────────────────────────────────────────────────
from integration_core.pipeline_manager import PipelineManager
from integration_core.api_interface    import start_server


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "main.py",
        description = textwrap.dedent("""\
            Smart Grid Cyber-Physical Attack Detection System
            Module 3 – System Architecture & Integration (Varun, Core CSE)
        """),
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )

    # Execution mode
    parser.add_argument(
        "--mode",
        choices = ["cli", "server"],
        default = "cli",
        help    = "Run pipeline in CLI mode or launch FastAPI server (default: cli).",
    )

    # Pipeline hyperparams (CLI mode)
    parser.add_argument(
        "--epochs",
        type    = int,
        default = 15,
        help    = "Training epochs for Module 2 detector (default: 15).",
    )
    parser.add_argument(
        "--temporal",
        choices = ["lstm", "transformer"],
        default = "lstm",
        help    = "Temporal encoder architecture (default: lstm).",
    )
    parser.add_argument(
        "--attack",
        default = "all",
        choices = ["all", "FDI", "spoofing", "manipulation",
                   "coordinated", "cascading"],
        help    = "Attack type to inject (default: all).",
    )
    parser.add_argument(
        "--gnn",
        action  = "store_true",
        help    = "Enable GNN layers (requires torch_geometric to be installed).",
    )
    parser.add_argument(
        "--export",
        action  = "store_true",
        help    = "Export raw Module 1 datasets to CSV as well.",
    )

    # Server mode options
    parser.add_argument(
        "--host",
        default = "0.0.0.0",
        help    = "Host for the FastAPI server (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type    = int,
        default = 8000,
        help    = "Port for the FastAPI server (default: 8000).",
    )

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# CLI mode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_cli(args: argparse.Namespace) -> None:
    """
    Execute the full M1 → M2 pipeline from the command line.
    Prints a rich summary table at the end and exits with code 0 on success.
    """
    _banner()
    print(f"  Mode        : CLI")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Temporal    : {args.temporal}")
    print(f"  Attack      : {args.attack}")
    print(f"  GNN         : {args.gnn}")
    print(f"  Export raw  : {args.export}")
    print(f"  Started     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    manager = PipelineManager(attack_filter=args.attack)

    results, ai_scores_df = manager.execute_pipeline(
        epochs        = args.epochs,
        temporal_type = args.temporal,
        use_gnn       = args.gnn,
        export_raw    = args.export,
    )

    # ── Print final summary ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(" PIPELINE SUMMARY ".center(65, "="))
    print("=" * 65)

    scores = results["anomaly_scores"]
    alerts = results["alerts"]

    print(f"\n  {'Metric':<30} {'Value':>20}")
    print("  " + "-" * 52)
    print(f"  {'Windows evaluated':<30} {len(scores):>20,}")
    print(f"  {'Mean anomaly score':<30} {scores.mean():>20.4f}")
    print(f"  {'Max anomaly score':<30} {scores.max():>20.4f}")
    print(f"  {'Alerts fired':<30} {int(alerts.sum()):>20,}")
    print(f"  {'Alert rate':<30} {alerts.mean()*100:>19.1f}%")

    if "test_results" in results and results["test_results"]:
        tr = results["test_results"]
        if isinstance(tr, dict):
            f1  = tr.get("f1",  tr.get("weighted avg", {}).get("f1-score", "N/A"))
            auc = tr.get("auc", "N/A")
            print(f"  {'F1-score (test)':<30} {str(f1):>20}")
            print(f"  {'ROC-AUC (test)':<30} {str(auc):>20}")

    print(f"\n  {'Output files':}")
    print(f"    data/processed/ai_anomaly_scores.csv  ← for Saanvi (M4)")
    print(f"    data/raw/scada_data.csv                ← raw SCADA")
    print(f"    results/model_report.json              ← classification report")
    print(f"    results/training_history.csv           ← loss curves")
    print(f"    results/pipeline_manifest.json         ← run manifest")
    print(f"    results/system_pipeline.log            ← full execution log")

    print("\n" + "=" * 65)
    print(" [HANDOFF READY] ai_anomaly_scores.csv → Saanvi / Module 4 ".center(65))
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner() -> None:
    print("=" * 65)
    print(" SMART GRID MASTER PIPELINE – INTEGRATION CORE (M3) ".center(65))
    print("=" * 65)
    print()
    print("  Signal → AI → [IDS] → [Fusion] → [Game Theory] → Output")
    print("  Module 3: System Architecture & Integration – Varun")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.mode == "server":
        # Launch FastAPI integration server
        _banner()
        print(f"  Mode  : Server")
        print(f"  URL   : http://{args.host}:{args.port}")
        print(f"  Docs  : http://localhost:{args.port}/docs")
        print()
        start_server(host=args.host, port=args.port)
    else:
        # Run pipeline end-to-end in CLI mode
        try:
            run_cli(args)
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Pipeline halted by user.")
            sys.exit(1)
        except Exception as exc:                      # noqa: BLE001
            print(f"\n[ERROR] Pipeline failed: {exc}")
            raise


if __name__ == "__main__":
    main()
