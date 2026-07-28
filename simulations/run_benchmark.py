"""Convenience wrapper; the installed CLI is the canonical entry point."""

from aging_clock_audit.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["benchmark"]))
