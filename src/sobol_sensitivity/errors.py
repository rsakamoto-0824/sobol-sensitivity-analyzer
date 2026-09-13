"""利用者向けのエラー"""


class SobolInputError(ValueError):
    """入力（CSV・コマンド引数・モデル関数）に問題があるときの例外。メッセージはそのまま画面に表示する"""
