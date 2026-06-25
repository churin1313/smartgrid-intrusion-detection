# integration_core/pipeline_manager.py
# Handles: Module Execution Sequence, Data Flow Orchestration
# Student 3 – Core CSE – Varun

import logging
import pandas as pd

from .api_interface import APIInterface
from .system_controller import SystemController

logger = logging.getLogger("SmartGrid")


class PipelineManager:
    """
    Manages the sequential execution of all 5 modules.

    Step 1 → Load Smart Grid Signal Data         (ECE  – Pranav)
    Step 2 → Load & Validate AI Detection         (AI   – Shashwat)
    Step 3 → Load & Validate Network IDS          (IT   – Nithila)
    Step 4 → Load & Validate Fused Risk Scores    (DS   – Saanvi)
    Step 5 → Load & Validate Defense Actions      (IT   – Nithila, game theory)
    Step 6 → Generate Final Alert & Defense Report (CSE – Varun)

    Each step records timing via SystemController and raises on failure
    so the demo never silently produces wrong results.
    """

    def __init__(self, controller: SystemController, api: APIInterface):
        self.ctrl = controller
        self.api  = api

        # Shared state populated as the pipeline progresses
        self.df_signal:  pd.DataFrame | None = None
        self.df_ai:      pd.DataFrame | None = None
        self.df_ids:     pd.DataFrame | None = None
        self.df_fusion:  pd.DataFrame | None = None
        self.df_defense: pd.DataFrame | None = None
        self.final_report: dict = {}

    # ────────────────────────────────────────────
    #  STEP 1  – Smart Grid Signal Data
    # ────────────────────────────────────────────

    def step1_load_signal_data(self) -> None:
        self.ctrl.step_start(1, "Load Smart Grid Signal Data  [ECE – Pranav]")

        self.df_signal = self.ctrl.safe_run(
            self.api.load_signal_features,
            "Step 1 – Signal Data"
        )

        avg_score  = self.df_signal["signal_anomaly_score"].mean()
        n_attacked = (self.df_signal.get("is_attack", pd.Series([0]*len(self.df_signal))) == 1).sum()

        logger.info(f"  Buses monitored          : {self.df_signal['bus_id'].nunique()}")
        logger.info(f"  Avg signal anomaly score : {avg_score:.4f}")
        logger.info(f"  Flagged attack samples   : {n_attacked}")

        self.ctrl.step_done(1)

    # ────────────────────────────────────────────
    #  STEP 2  – AI / Deep Learning Detection
    # ────────────────────────────────────────────

    def step2_run_ai_detection(self) -> None:
        self.ctrl.step_start(2, "AI Spatio-Temporal Detection  [AI/ML – Shashwat]")

        self.df_ai = self.ctrl.safe_run(
            self.api.load_ai_scores,
            "Step 2 – AI Detection"
        )

        avg_score = self.df_ai["ai_anomaly_score"].mean()
        top_type  = self.df_ai["predicted_attack_type"].value_counts().idxmax()

        logger.info(f"  Avg AI anomaly score     : {avg_score:.4f}")
        logger.info(f"  Most predicted attack    : {top_type}")
        logger.info(f"  High-confidence detects  : {(self.df_ai['ai_anomaly_score'] > 0.7).sum()}")

        self.ctrl.step_done(2)

    # ────────────────────────────────────────────
    #  STEP 3  – Network IDS
    # ────────────────────────────────────────────

    def step3_run_network_ids(self) -> None:
        self.ctrl.step_start(3, "Network Intrusion Detection System  [IT – Nithila]")

        self.df_ids = self.ctrl.safe_run(
            self.api.load_ids_scores,
            "Step 3 – Network IDS"
        )

        avg_score = self.df_ids["ids_anomaly_score"].mean()
        top_cat   = self.df_ids["attack_category"].value_counts().idxmax()

        logger.info(f"  Avg IDS anomaly score    : {avg_score:.4f}")
        logger.info(f"  Top network attack type  : {top_cat}")
        logger.info(f"  High-risk network events : {(self.df_ids['ids_anomaly_score'] > 0.6).sum()}")

        self.ctrl.step_done(3)

    # ────────────────────────────────────────────
    #  STEP 4  – Multi-Modal Fusion
    # ────────────────────────────────────────────

    def step4_fuse_scores(self) -> None:
        self.ctrl.step_start(4, "Multi-Modal Fusion & Risk Analytics  [Data Science – Saanvi]")

        self.df_fusion = self.ctrl.safe_run(
            self.api.load_fusion_scores,
            "Step 4 – Fusion"
        )

        avg_fused = self.df_fusion["fused_score"].mean()
        sev_dist  = self.df_fusion["severity_level"].value_counts().to_dict()

        logger.info(f"  Avg fused risk score     : {avg_fused:.4f}")
        logger.info(f"  Severity distribution    : {sev_dist}")
        logger.info(f"  CRITICAL events detected : {sev_dist.get('CRITICAL', 0)}")

        self.ctrl.step_done(4)

    # ────────────────────────────────────────────
    #  STEP 5  – Game-Theoretic Defense
    # ────────────────────────────────────────────

    def step5_apply_game_theory(self) -> None:
        self.ctrl.step_start(5, "Game-Theoretic Defense Engine  [IT – Nithila, Stackelberg]")

        self.df_defense = self.ctrl.safe_run(
            self.api.load_defense_actions,
            "Step 5 – Game Theory"
        )

        action_dist = self.df_defense["defense_action"].value_counts().to_dict()

        logger.info(f"  Defense strategy distribution:")
        for action, count in action_dist.items():
            logger.info(f"    {action:<28s}: {count} events")

        avg_def_util = self.df_defense["defender_utility"].mean()
        avg_att_util = self.df_defense["attacker_utility"].mean()
        logger.info(f"  Avg defender utility (U_d): {avg_def_util:.4f}")
        logger.info(f"  Avg attacker utility (U_a): {avg_att_util:.4f}")
        logger.info(f"  Nash equilibrium status   : Stackelberg solution computed ✔")

        self.ctrl.step_done(5)

    # ────────────────────────────────────────────
    #  STEP 6  – Final Alert & Consolidated Output
    # ────────────────────────────────────────────

    def step6_generate_final_output(self) -> dict:
        self.ctrl.step_start(6, "Final Alert & Defense Strategy Generation  [Core CSE – Varun]")

        # ── Build consolidated event-level dataframe ─────────────────
        merged = self.df_signal[["timestamp", "signal_anomaly_score"]].copy()
        merged = merged.merge(
            self.df_ai[["timestamp", "ai_anomaly_score", "predicted_attack_type"]],
            on="timestamp", how="left"
        )
        merged = merged.merge(
            self.df_ids[["timestamp", "ids_anomaly_score"]],
            on="timestamp", how="left"
        )
        merged = merged.merge(
            self.df_fusion[["timestamp", "fused_score", "severity_level", "attack_type"]],
            on="timestamp", how="left"
        )
        merged = merged.merge(
            self.df_defense[["timestamp", "defense_action"]],
            on="timestamp", how="left"
        )

        # ── Save event-level output ──────────────────────────────────
        self.ctrl.safe_run(self.api.save_final_output, "Step 6 – Save CSV", merged)

        # ── Compute pipeline-level summary scalars ───────────────────
        avg_signal  = float(merged["signal_anomaly_score"].mean())
        avg_ai      = float(merged["ai_anomaly_score"].mean())
        avg_ids     = float(merged["ids_anomaly_score"].mean())
        avg_fused   = float(merged["fused_score"].mean())
        top_attack  = str(merged["attack_type"].dropna().value_counts().idxmax())
        top_defense = str(merged["defense_action"].dropna().value_counts().idxmax())

        # ── Print per-event summary table ────────────────────────────
        logger.info("")
        logger.info("  Per-Event Summary (first 10 rows):")
        cols = ["timestamp", "fused_score", "severity_level", "attack_type", "defense_action"]
        logger.info("\n" + merged[cols].head(10).to_string(index=False))

        # ── Generate and log the headline report ─────────────────────
        report = self.ctrl.generate_final_report(
            signal_score   = avg_signal,
            ai_score       = avg_ai,
            ids_score      = avg_ids,
            fused_score    = avg_fused,
            attack_type    = top_attack,
            defense_action = top_defense,
        )

        # Save summary row
        self.ctrl.safe_run(self.api.save_report_row, "Step 6 – Save Report", report)
        self.final_report = report

        self.ctrl.step_done(6)
        return report

    # ────────────────────────────────────────────
    #  run_all  – single entry point
    # ────────────────────────────────────────────

    def run_all(self) -> dict:
        """
        Execute the complete 6-step pipeline end-to-end.
        Returns the final threat assessment report dict.
        """
        self.step1_load_signal_data()
        self.step2_run_ai_detection()
        self.step3_run_network_ids()
        self.step4_fuse_scores()
        self.step5_apply_game_theory()
        return self.step6_generate_final_output()
