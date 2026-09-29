"""Stage 2 Rhythm2 訓練入口：與 main_stage2realfake_rhythm.py 共用整套流程，只換模型。

模型為 ``ProSDDStage2Rhythm2``（ProSDD backbone 加 rhythm-transformer 融合層），
參數、資料流程、optimizer 分組、紀錄與 checkpoint 行為完全相同；
``config.json`` 的 ``model_class`` 會記為 ``ProSDDStage2Rhythm2`` 供評估入口還原。
"""

import main_stage2realfake_rhythm as rhythm
from model_stage2realfake_rhythm2 import ProSDDStage2Rhythm2


def main(argv=None):
    """以相同 CLI 參數執行 Rhythm 訓練流程，模型改用 ProSDDStage2Rhythm2。"""
    return rhythm.main(argv, model_cls=ProSDDStage2Rhythm2)


if __name__ == "__main__":
    main()
