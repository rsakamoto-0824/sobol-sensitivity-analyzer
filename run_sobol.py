"""インストールせずに実行するための起動スクリプト（例: python run_sobol.py check）"""

import sys
from pathlib import Path

SOURCE_DIRECTORY = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from sobol_sensitivity.cli import main  # noqa: E402  上で読み込み先を追加してから読み込む

if __name__ == "__main__":
    sys.exit(main())
