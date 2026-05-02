from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass
class SkillDeployConfig:
    skill_input_mode: Literal["fixed", "operator", "external"] = "fixed"
    default_skill: list[float] | None = None


@dataclass
class FactorConfig:
    name: str
    method: Literal["diayn", "metra"]
    skill_dim: int
    source: Literal["actor_obs", "critic_obs", "sim_data", "info"]
    extractor: str
    dim: int


@dataclass
class SkillDiscoveryConfig:
    enabled: bool = True
    mode: Literal["diayn", "metra", "factorized"] = "diayn"
    total_skill_dim: int = 0
    prior: Literal["one_hot", "dirichlet", "hypersphere"] = "one_hot"
    resample: Literal["episode", "fixed_steps"] = "episode"
    resample_steps: int | None = None
    reward_mode: Literal["sample_time", "collection_time"] = "sample_time"
    intrinsic_weight: float = 1.0
    task_reward_weight: float = 0.0
    style_reward_weight: float = 0.0
    safety_penalty_weight: float = 0.0
    factors: tuple[FactorConfig, ...] = ()
    deploy: SkillDeployConfig = field(default_factory=SkillDeployConfig)

    def __post_init__(self):
        if self.factors:
            expected = sum(f.skill_dim for f in self.factors)
            if self.total_skill_dim != expected:
                raise ValueError(
                    f"total_skill_dim ({self.total_skill_dim}) must equal "
                    f"sum of factor skill_dim ({expected})"
                )
        if self.resample == "fixed_steps" and self.resample_steps is None:
            raise ValueError("resample_steps must be set when resample == 'fixed_steps'")


def config_to_dict(cfg: SkillDiscoveryConfig) -> dict:
    d = asdict(cfg)
    d["factors"] = list(d["factors"])
    return d


def config_from_dict(d: dict) -> SkillDiscoveryConfig:
    kwargs = dict(d)
    kwargs["factors"] = tuple(FactorConfig(**f) for f in kwargs.get("factors", []))
    if "deploy" in kwargs and kwargs["deploy"] is not None:
        kwargs["deploy"] = SkillDeployConfig(**kwargs["deploy"])
    return SkillDiscoveryConfig(**kwargs)
