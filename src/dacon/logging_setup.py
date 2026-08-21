"""로깅 설정: tqdm 진행률 막대와 로그가 겹치지 않게 한 곳에서 구성한다.

`logging.info(...)` 한 방식만 쓰면서, 출력은 `tqdm.write()`를 거치게 해
진행률 막대가 그려지는 중간에 로그가 끼어들어도 줄이 깨지지 않는다.
"""
import logging

from tqdm.auto import tqdm


class TqdmLoggingHandler(logging.Handler):
    """로그를 `tqdm.write()`로 내보내 진행률 막대와 충돌하지 않게 하는 핸들러."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:  # noqa: BLE001 - 로깅이 파이프라인을 멈추면 안 된다
            self.handleError(record)


def setup_logging(level: int = logging.INFO) -> None:
    """루트 로거를 tqdm 친화 핸들러 하나로 재구성한다(중복 핸들러 방지)."""
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
