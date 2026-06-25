#!/usr/bin/env python3
"""
main.py – Final System Pipeline
================================
Game-Theoretic Multi-Model Cyber-Physical Attack Detection Framework
for Smart Grid

Author  : Student 3 – Core CSE – Varun (25BCE2455)
Guide   : Dr. Priya V, Sr Associate Professor, SCORE – VIT
Role    : System Integrator & Technical Lead

Pipeline Flow
-------------
Step 1 → Load Smart Grid Signal Data       (ECE  – Pranav)
Step 2 → Run AI Spatio-Temporal Detection  (AI   – Shashwat)
Step 3 → Run Network IDS                   (IT   – Nithila)
Step 4 → Fuse Multi-Modal Scores           (DS   – Saanvi)
Step 5 → Apply Game-Theoretic Defense      (IT   – Nithila, Stackelberg)
Step 6 → Generate Final Alert & Strategy   (CSE  – Varun)
"""

import os
import sys

# ── Make sure the project root is on the Python path ──────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Core imports ──────────────────────────────────────────────────────────────
from integration_core.system_controller import SystemController
from integration_core.api_interface import APIInterface, initialise_mock_data
from integration_core.pipeline_manager import PipelineManager


def banner(ctrl) -> None:
    lines = [
        "",
        "╔══════════════════════════════════════════════════════════════╗",
        "║   SmartGrid Cyber-Physical Attack Detection Framework        ║",
        "║   Game-Theoretic Multi-Model IDS  –  VIT SCORE               ║",
        "║   Student 3: System Integration Core  |  Varun (25BCE2455)  ║",
        "╚══════════════════════════════════════════════════════════════╝",
        "",
    ]
    for line in lines:
        ctrl.logger.info(line)


def main() -> None:
    # ── 1. Bootstrap ──────────────────────────────────────────────────────────
    controller = SystemController(log_dir=os.path.join("results", "logs"))
    banner(controller)
    controller.logger.info("Bootstrapping pipeline…")

    # ── 2. Generate mock data for missing teammate outputs ─────────────────────
    #       If a teammate's CSV already exists in data/processed/,
    #       initialise_mock_data() will NOT overwrite it.
    initialise_mock_data()

    # ── 3. Create API interface and pipeline manager ───────────────────────────
    api      = APIInterface()
    pipeline = PipelineManager(controller=controller, api=api)

    # ── 4. Run the full 6-step pipeline ───────────────────────────────────────
    controller.logger.info("Launching 6-step execution pipeline…")
    report = pipeline.run_all()

    # ── 5. Final status ────────────────────────────────────────────────────────
    controller.logger.info(
        f"Pipeline complete. "
        f"Threat level: {report['severity_level']} | "
        f"Action: {report['defense_action']} | "
        f"Resilience: {report['resilience_score']}"
    )
    controller.logger.info("Results saved to → results/final_outputs.csv")
    controller.logger.info("Logs saved to    → results/logs/")


if __name__ == "__main__":
    main()
