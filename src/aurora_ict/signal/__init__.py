"""Aurora-ICT signal — bot이 소비하는 signal layer.

strategy가 만든 setup 중 진입 가능한 것을 signal로 변환한다.
bot_ict_instance가 이 signal을 받아 주문을 실행한다.
"""

from aurora_ict.signal.ict_signal import (
    ICTSignal,
    SignalAction,
    generate_ict_signal,
)

__all__ = ["ICTSignal", "SignalAction", "generate_ict_signal"]
