"""
=============================================================================
integration_core — Module 3: System Architecture, Backend & Integration
=============================================================================
Student 3 – Core CSE – Varun
Project : Game-Theoretic Multi-Model Cyber-Physical Attack Detection for
          Smart Grid (Dr. Priya V, SCORE, VIT)

Exposes three sub-components:
    system_controller  – directory management, logging, file I/O
    pipeline_manager   – M1 → M2 orchestration, CSV hand-off to Saanvi
    api_interface      – FastAPI REST layer for real-time triggering
=============================================================================
"""

from .system_controller import SystemController
from .pipeline_manager  import PipelineManager

__all__ = ["SystemController", "PipelineManager"]
__version__ = "1.0.0"
