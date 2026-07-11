"""Command line interface for the AI Aquatica package."""

from __future__ import annotations

import argparse
import inspect
import sys
from importlib import metadata
from typing import Iterable, Tuple


def _get_version() -> str:
    """Return the installed package version."""

    try:
        return metadata.version("AI-Aquatica")
    except metadata.PackageNotFoundError:  # pragma: no cover - fallback for editable installs
        return "unknown"


def _collect_public_api() -> Tuple[Iterable[str], Iterable[Tuple[str, str]]]:
    """Collect modules and callables exported via :data:`ai_aquatica.__all__`."""

    import ai_aquatica

    modules = []
    callables = []
    for name in getattr(ai_aquatica, "__all__", ()):  # pragma: no cover - defensive guard
        attr = getattr(ai_aquatica, name, None)
        if inspect.ismodule(attr):
            modules.append(name)
        elif callable(attr):
            doc = inspect.getdoc(attr) or ""
            callables.append((name, doc.splitlines()[0] if doc else ""))

    modules.sort()
    callables.sort(key=lambda item: item[0])
    return modules, callables


def _describe_symbol(symbol: str) -> Tuple[str, str]:
    """Return the first line of the docstring for a public symbol."""

    import ai_aquatica

    if symbol not in getattr(ai_aquatica, "__all__", ()):  # pragma: no cover - defensive guard
        raise AttributeError(symbol)

    attr = getattr(ai_aquatica, symbol)
    doc = inspect.getdoc(attr) or ""
    headline = doc.splitlines()[0] if doc else "(no documentation available)"
    return symbol, headline


def main(argv: Iterable[str] | None = None) -> int:
    """Entry-point for the ``ai-aquatica`` console script."""

    parser = argparse.ArgumentParser(description="Utilities for exploring AI Aquatica's features.")
    parser.add_argument(
        "--version",
        action="store_true",
        help="print the installed package version",
    )
    parser.add_argument(
        "--list-exports",
        action="store_true",
        help="list the public modules and functions exported by the package",
    )
    parser.add_argument(
        "--describe",
        metavar="SYMBOL",
        help="show the first line of documentation for a public symbol",
    )

    args = parser.parse_args(argv)

    exit_code = 0
    handled = False

    if args.version:
        print(f"AI-Aquatica { _get_version() }")
        handled = True

    if args.list_exports:
        modules, callables = _collect_public_api()
        print("Public modules:")
        for name in modules:
            print(f"  - {name}")

        print("\nPublic functions:")
        for name, headline in callables:
            summary = f" — {headline}" if headline else ""
            print(f"  - {name}{summary}")

        if not modules and not callables:
            print("  (no public exports discovered)")
        handled = True

    if args.describe:
        try:
            name, headline = _describe_symbol(args.describe)
        except AttributeError:
            print(f"Error: '{args.describe}' is not part of the public API", file=sys.stderr)
            exit_code = 1
        else:
            print(f"{name}: {headline}")
        handled = True

    if not handled:
        parser.print_help()

    return exit_code


if __name__ == "__main__":  # pragma: no cover - manual invocation convenience
    sys.exit(main())
