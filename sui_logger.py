"""
Minimal helper to write a Move event on Sui devnet each time the chatbot
answers a question. Requires `sui` Python SDK.

Move package (pre-published by sponsors) :
    0xmarketpulse::querylog::log(question_hash, answer_hash, ts)

If that package isn’t published you can still log with `sui_object::note`.
"""
import os, hashlib, time, logging
from typing import Optional

try:
    from sui import SuiClient, sui_types
except ImportError:  # noop if lib not installed
    SuiClient = None

LOG = logging.getLogger("sui_logger")

_RPC = os.getenv("SUI_RPC", "https://fullnode.devnet.sui.io:443")
_ADDR = os.getenv("SUI_ADDRESS")
_PRIV = os.getenv("SUI_PRIVATE_KEY")
_ENABLED = os.getenv("ENABLE_SUI_LOGGER", "false").lower() == "true" and _ADDR and _PRIV

_client: Optional[SuiClient] = None
if _ENABLED and SuiClient:
    _client = SuiClient(_RPC, _ADDR, _PRIV)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def log_interaction(question: str, answer: str, ticker: str | None = None) -> None:
    """
    Fire-and-forget: hashes the Q&A, sends a Move call.
    No raise on failure – just log.
    """
    if not _client:
        return

    try:
        txb = _client.new_transaction()
        txb.move_call(
            target="0xmarketpulse::querylog::log",
            arguments=[
                _sha256(question),
                _sha256(answer),
                int(time.time()),
                ticker or "N/A",
            ],
        )
        txb.set_gas_budget(1_000_000)
        res = _client.execute(txb)
        LOG.debug("Sui log tx: %s", res)
    except Exception as e:
        LOG.error("Sui log failed: %s", e)
