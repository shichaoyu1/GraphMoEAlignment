"""Compatibility entry point; the project is now named Idea4.5."""

from glioma.cli.idea4_5 import build_parser, main


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "main"]
