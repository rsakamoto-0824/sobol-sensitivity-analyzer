"""python -m sobol_sensitivity で実行するための入口"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
