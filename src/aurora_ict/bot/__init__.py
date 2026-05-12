"""Aurora-ICT bot — 봇 instance와 실제 매매 실행 layer."""

from aurora_ict.bot.aurora_adapter import AuroraClientAdapter
from aurora_ict.bot.bot_ict_instance import (
    BotIctInstance,
    BotState,
    ExchangeClientProtocol,
)
from aurora_ict.bot.manager import BotManager, BotStatus, ClientFactory

__all__ = [
    "AuroraClientAdapter",
    "BotIctInstance",
    "BotManager",
    "BotState",
    "BotStatus",
    "ClientFactory",
    "ExchangeClientProtocol",
    "aurora_client_factory",
]


async def aurora_client_factory(settings):  # type: ignore[no-untyped-def]
    """Aurora ``CcxtClient``를 생성해 어댑터로 감싸 반환하는 factory.

    settings의 run_mode / API 키 / demo 여부를 그대로 사용한다.
    """
    from aurora.exchange.ccxt_client import CcxtClient  # type: ignore[import-not-found]
    client = CcxtClient(
        exchange_id="bybit",
        api_key=settings.active_api_key,
        api_secret=settings.active_api_secret,
        demo=settings.is_demo,
    )
    return AuroraClientAdapter(client)
