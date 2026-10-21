from pathlib import Path

import polars as pl

from loader import load_trades



def main():

    df = load_trades()


if __name__ == "__main__":
    main()