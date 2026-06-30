"""Multi-physics degradation models combined into a Transitory Deterioration Rate.

* ``arrhenius``      — thermal stress
* ``hallberg_peck``  — temperature/humidity stress
* ``hsu_soiling``    — particulate soiling with rain cleaning
* ``tdr``            — aggregates the above with LID and linear ageing
"""

from solarpro.degradation.arrhenius import arrhenius_acceleration
from solarpro.degradation.hallberg_peck import hallberg_peck_acceleration
from solarpro.degradation.hsu_soiling import hsu_soiling_ratio
from solarpro.degradation.tdr import transitory_deterioration_rate

__all__ = [
    "arrhenius_acceleration",
    "hallberg_peck_acceleration",
    "hsu_soiling_ratio",
    "transitory_deterioration_rate",
]
