"""Command-line scaffold; research pipelines are not implemented yet."""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print("Project scaffold ready. Research pipelines are not implemented yet.")
