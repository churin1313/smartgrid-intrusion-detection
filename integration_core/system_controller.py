# integration_core/system_controller.py
# Handles: Logging, Error Handling, Final Output Generation
# Student 3 – Core CSE – Varun

import logging
import os
import sys
from datetime import datetime

try:
    import colorlog
    _HAS_COLOR = True
except ImportError:
    _HAS_COLOR = False


# ──────────────────────────────────────────────
#  Logger Factory
# ──────────────────────────────────────────────

def build_logger(name: str = "SmartGrid", log_dir: str = "results/logs") -> logging.Logger:
    """
    Creates and returns a logger that writes to both the console (coloured)
    and a timestamped log file under results/logs/.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers on re-import
    if logger.handlers:
        return logger

    fmt_str = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%H:%M:%S"

    # ── Console handler (coloured if colorlog is available) ──
    if _HAS_COLOR:
        colour_fmt = (
            "%(log_color)s%(asctime)s  %(levelname)-8s%(reset)s  %(message)s"
        )
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            colorlog.ColoredFormatter(
                colour_fmt,
                datefmt=datefmt,
                log_colors={
                    "DEBUG":    "cyan",
                    "INFO":     "green",
                    "WARNING":  "yellow",
                    "ERROR":    "red",
                    "CRITICAL": "bold_red",
                },
            )
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(fmt_str, datefmt=datefmt))

    console_handler.setLevel(logging.INFO)

    # ── File handler (plain text) ──
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(fmt_str, datefmt=datefmt))
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# ──────────────────────────────────────────────
#  SystemController
# ──────────────────────────────────────────────

class SystemController:
    """
    Central orchestration brain.

    Responsibilities
    ----------------
    * Hold the shared logger used across all modules.
    * Track pipeline step execution with timing.
    * Generate the final human-readable threat report.
    * Handle top-level errors gracefully so the demo never crashes.
    """

    SEVERITY_THRESHOLDS = {
        "LOW":      (0.00, 0.35),
        "MEDIUM":   (0.35, 0.65),
        "HIGH":     (0.65, 0.85),
        "CRITICAL": (0.85, 1.01),
    }

    DEFENSE_COST = {
        "Monitor":           0.10,
        "Isolate Node":      0.45,
        "Reconfigure Grid":  0.70,
        "Trigger Alert":     0.30,
        "Secure Re-Estimation": 0.55,
    }

    def __init__(self, log_dir: str = "results/logs"):
        self.logger = build_logger(log_dir=log_dir)
        self._step_times: dict = {}
        self._start_ts = datetime.now()

    # ── Step tracking ──────────────────────────

    def step_start(self, step_num: int, description: str) -> None:
        self._step_times[step_num] = datetime.now()
        bar = "─" * 60
        self.logger.info(bar)
        self.logger.info(f"STEP {step_num}  ▶  {description}")
        self.logger.info(bar)

    def step_done(self, step_num: int) -> None:
        elapsed = (datetime.now() - self._step_times[step_num]).total_seconds()
        self.logger.info(f"STEP {step_num}  ✔  Completed in {elapsed:.3f}s")

    # ── Severity classifier ────────────────────

    def classify_severity(self, fused_score: float) -> str:
        for label, (lo, hi) in self.SEVERITY_THRESHOLDS.items():
            if lo <= fused_score < hi:
                return label
        return "CRITICAL"

    # ── Resilience score ───────────────────────

    def compute_resilience_score(
        self,
        fused_score: float,
        defense_action: str
    ) -> float:
        """
        Resilience = 1 – (threat_weight × unmitigated_fraction)

        A stronger (costlier) defense action reduces residual threat more.
        """
        defense_strength = self.DEFENSE_COST.get(defense_action, 0.3)
        residual_threat   = fused_score * (1 - defense_strength)
        resilience        = round(max(0.0, 1.0 - residual_threat), 4)
        return resilience

    # ── Final report ───────────────────────────

    def generate_final_report(
        self,
        signal_score:  float,
        ai_score:      float,
        ids_score:     float,
        fused_score:   float,
        attack_type:   str,
        defense_action: str,
    ) -> dict:
        """
        Builds and logs the complete threat + defense summary.
        Returns a dict suitable for JSON / CSV export.
        """
        severity        = self.classify_severity(fused_score)
        resilience      = self.compute_resilience_score(fused_score, defense_action)
        total_elapsed   = (datetime.now() - self._start_ts).total_seconds()

        report = {
            "timestamp":       datetime.now().isoformat(timespec="seconds"),
            "signal_score":    round(signal_score,  4),
            "ai_score":        round(ai_score,       4),
            "ids_score":       round(ids_score,      4),
            "fused_score":     round(fused_score,    4),
            "attack_type":     attack_type,
            "severity_level":  severity,
            "defense_action":  defense_action,
            "resilience_score": resilience,
            "pipeline_runtime_s": round(total_elapsed, 3),
        }

        border = "═" * 65
        self.logger.info("")
        self.logger.info(border)
        self.logger.info("   FINAL THREAT ASSESSMENT REPORT")
        self.logger.info(border)
        self.logger.info(f"  Timestamp         : {report['timestamp']}")
        self.logger.info(f"  Signal Score      : {report['signal_score']}")
        self.logger.info(f"  AI Score          : {report['ai_score']}")
        self.logger.info(f"  IDS Score         : {report['ids_score']}")
        self.logger.info(f"  ── FUSED SCORE ── : {report['fused_score']}")
        self.logger.info(f"  Attack Type       : {report['attack_type']}")
        self.logger.info(f"  Severity Level    : {report['severity_level']}")
        self.logger.info(f"  Defense Action    : {report['defense_action']}")
        self.logger.info(f"  Resilience Score  : {report['resilience_score']}")
        self.logger.info(f"  Pipeline Runtime  : {report['pipeline_runtime_s']}s")
        self.logger.info(border)
        self.logger.info("")

        return report

    # ── Error guard ────────────────────────────

    def safe_run(self, func, step_name: str, *args, **kwargs):
        """
        Wraps any callable in a try/except so a single module failure
        does not bring down the whole demo.
        """
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            self.logger.error(f"[{step_name}] FAILED → {exc}")
            raise SystemExit(1) from exc
