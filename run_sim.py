import argparse
from typing import Any, Dict, List, Tuple, Union
from simulator import Simulator


def parse_arguments() -> Any:
    """Arguments for running the baseline.

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, default="", type=str)
    return parser.parse_args()


def main():
    args = parse_arguments()
    simulator = Simulator(args.config)
    simulator.run()

if __name__ == "__main__":
    main()
