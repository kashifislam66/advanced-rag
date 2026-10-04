import logging


def setup_logging(level: str = "INFO") -> None:
    """Configure application-wide logging once at startup."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )