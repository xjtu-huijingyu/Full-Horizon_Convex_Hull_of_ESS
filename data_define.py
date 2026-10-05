from dataclasses import dataclass

@dataclass(slots=True)
class Storage:
    """ESS unit"""
    id: int = 0
    emax: float = 0.0  # Maximum energy capacity
    emin: float = 0.0  # Minimum energy capacity
    pch_max: float = 0.0  # Maximum charging power
    pch_min: float = 0.0  # Minimum charging power
    pdis_max: float = 0.0  # Maximum discharging power
    pdis_min: float = 0.0  # Minimum discharging power
    eff_ch: float = 0.0  # Charging efficiency
    eff_dis: float = 0.0  # Discharging efficiency
    Ldis: int = 0  # Minimum discharging duration; reserved and unused in the Pozo formulation
    Lch: int = 0  # Minimum charging duration; reserved and unused in the Pozo formulation
    Lidle: int = 0  # Minimum idle duration; reserved and unused in the Pozo formulation
    # Initial ESS state; reserved and unused in the Pozo formulation
    init_e: float = 0.0  # Initial energy level
    init_xdis: int = 0  # Initial discharging state
    init_xch: int = 0  # Initial charging state
    init_xidle: int = 1  # Initial idle state
    init_duration: int = 0  # Initial-state duration
    end_e: float = 0.0  # Terminal energy level; reserved and unused in the Pozo formulation
