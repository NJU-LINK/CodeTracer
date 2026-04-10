"""PluginAdapter: base class for integrating CodeTracer into external agent frameworks.

Each adapter translates between a framework's native trajectory format and
CodeTracer's NormalizedTrajectory, then exposes analyze / replay as simple calls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from codetracer.models import ErrorAnalysis, NormalizedTrajectory, ReplayResult


class PluginAdapter(ABC):
    """Abstract integration surface for embedding CodeTracer in another agent framework."""

    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this adapter (e.g. 'openhands', 'swe_agent')."""

    @abstractmethod
    def ingest_trajectory(self, raw_path: Path, **kwargs: Any) -> NormalizedTrajectory:
        """Convert the framework's native trajectory output into a NormalizedTrajectory."""

    @abstractmethod
    def analyze(self, traj: NormalizedTrajectory, **kwargs: Any) -> ErrorAnalysis:
        """Run CodeTracer error analysis on the normalized trajectory."""

    @abstractmethod
    def replay(
        self,
        traj: NormalizedTrajectory,
        step_id: int,
        analysis: ErrorAnalysis,
        **kwargs: Any,
    ) -> ReplayResult:
        """Replay the trajectory to *step_id* with error analysis injected."""

    def analyze_and_replay(self, raw_path: Path, **kwargs: Any) -> ReplayResult:
        """Convenience: ingest -> analyze -> auto-replay in one call."""
        traj = self.ingest_trajectory(raw_path, **kwargs)
        analysis = self.analyze(traj, **kwargs)
        target = analysis.first_incorrect_step_id
        if target is None:
            from codetracer.models import ReplayStatus
            return ReplayResult(status=ReplayStatus.SUCCESS, steps_replayed=0)
        return self.replay(traj, target, analysis, **kwargs)


# ---------------------------------------------------------------------------
# Generic adapter: wires SkillPool + Normalizer + TraceAgent + ReplayEngine
# ---------------------------------------------------------------------------


class GenericPluginAdapter(PluginAdapter):
    """Full pipeline adapter backed by existing CodeTracer components.

    Subclass and override ``name()`` + ``__init__`` with the right
    ``skill_name`` to get a working adapter for any supported format.
    """

    def __init__(
        self,
        skill_name: str,
        *,
        bench_name: str | None = None,
        config: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._skill_name = skill_name
        self._bench_name = bench_name
        self._config: dict[str, Any] = config or {}
        self._llm_kwargs: dict[str, Any] = llm_kwargs or {}

    def name(self) -> str:
        return self._skill_name

    # -- ingest ---------------------------------------------------------------

    def ingest_trajectory(self, raw_path: Path, **kwargs: Any) -> NormalizedTrajectory:
        from codetracer.query.normalizer import Normalizer
        from codetracer.skills.pool import SkillPool

        pool = SkillPool()
        normalizer = Normalizer(pool)

        if normalizer.is_pre_normalized(raw_path):
            return normalizer.normalize_pre_normalized(raw_path)
        if normalizer.is_step_jsonl_dir(raw_path):
            return normalizer.normalize_step_jsonl(raw_path)

        skill = pool.get(self._skill_name) or normalizer.detect(raw_path)
        return normalizer.normalize(raw_path, skill)

    # -- analyze --------------------------------------------------------------

    def analyze(self, traj: NormalizedTrajectory, **kwargs: Any) -> ErrorAnalysis:
        run_dir = Path(kwargs.get("run_dir", ".")).resolve()
        output_path = run_dir / "codetracer_labels.json"

        from codetracer.agents.context import ContextAssembler
        from codetracer.agents.trace_agent import TraceAgent
        from codetracer.query.tree_builder import TreeBuilder
        from codetracer.skills.pool import SkillPool

        tree_path = run_dir / "tree.md"
        if not tree_path.exists():
            tree_path.write_text(TreeBuilder().build(traj), encoding="utf-8")

        llm = self._make_llm(**kwargs)
        pool = SkillPool()
        assembler = ContextAssembler(self._config, pool)
        agent = TraceAgent(llm, assembler, run_dir, output_path, self._config,
                           agent_type=self._skill_name or "")
        skill = pool.get(self._skill_name)
        agent.run(skill)

        return self._load_analysis(output_path, run_dir.name)

    # -- replay ---------------------------------------------------------------

    def replay(
        self,
        traj: NormalizedTrajectory,
        step_id: int,
        analysis: ErrorAnalysis,
        **kwargs: Any,
    ) -> ReplayResult:
        from codetracer.replay.engine import ReplayEngine

        run_dir = Path(kwargs.get("run_dir", ".")).resolve()
        engine = ReplayEngine(
            config=self._config,
            checkpoints_dir=run_dir / ".checkpoints",
        )
        llm = self._make_llm(**kwargs)
        return engine.replay_interactive(
            traj, step_id, analysis,
            env_config={"work_dir": str(run_dir)},
            llm=llm,
        )

    # -- helpers --------------------------------------------------------------

    def _make_llm(self, **kwargs: Any) -> Any:
        from codetracer.llm.client import LLMClient

        merged = dict(self._llm_kwargs)
        if "model" in kwargs:
            merged["model_name"] = kwargs["model"]
        if "api_base" in kwargs:
            merged["api_base"] = kwargs["api_base"]
        if "api_key" in kwargs:
            merged["api_key"] = kwargs["api_key"]
        return LLMClient(**merged)

    @staticmethod
    def _load_analysis(labels_path: Path, traj_id: str) -> ErrorAnalysis:
        if not labels_path.exists():
            return ErrorAnalysis(traj_id=traj_id)
        return ErrorAnalysis.from_labels_json(labels_path, traj_id)


# ---------------------------------------------------------------------------
# Concrete thin-wrapper adapters
# ---------------------------------------------------------------------------


class MinisweAdapter(GenericPluginAdapter):
    """Adapter for MiniSWE agent framework."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(skill_name="miniswe", **kwargs)

    def name(self) -> str:
        return "miniswe"


class OpenHandsAdapter(GenericPluginAdapter):
    """Adapter for OpenHands agent framework."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(skill_name="openhands", **kwargs)

    def name(self) -> str:
        return "openhands"


class SweAgentAdapter(GenericPluginAdapter):
    """Adapter for SWE-Agent framework (uses openhands skill with auto-detect fallback)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(skill_name="swe_agent", **kwargs)

    def name(self) -> str:
        return "swe_agent"

    def ingest_trajectory(self, raw_path: Path, **kwargs: Any) -> NormalizedTrajectory:
        from codetracer.query.normalizer import Normalizer
        from codetracer.skills.pool import SkillPool

        pool = SkillPool()
        normalizer = Normalizer(pool)

        skill = pool.get("swe_agent")
        if skill is None:
            skill = pool.get("openhands") or normalizer.detect(raw_path)
        return normalizer.normalize(raw_path, skill)
