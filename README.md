# Scaler Mimic Demo

循环播放展示脚本正在使用的 **43100** 策略。固定 160 格 TRAIN 得分 **[34, 123, 39, 136]**（完整且六门通过、任务通过、采样安全、完整完成）。这之后没有新的同协议分数超过这一行，所以演示就用这次快照。

单次 TRAIN 开发集结果，不是 heldout，也不是正式验收。

## 一键查看

```bash
git clone git@github.com:wangshuyiHIT/scaler_mimic_demo.git && cd scaler_mimic_demo && bash demo.sh
```

第一次运行会自建 `.venv`，安装 CPU 版 PyTorch 2.7.1 和 MuJoCo 3.8.1，然后打开窗口。已有环境时：

```bash
SCALER_MIMIC_PYTHON=/path/to/python bash demo.sh
```

`bash scripts/sim2sim.sh --loop` 是同一条入口。

| 键 | 作用 |
| --- | --- |
| 左 / 右 | 上一条 / 下一条动作 |
| 按住上 / 下 | 连续变高 / 变矮，松开即停 |
| 空格 | 暂停 / 继续 |
| R | 从头播放 |
| L | 切换循环 |
| Esc | 退出 |

策略在 CPU 上推理。窗口里的身高变化只改当前播放的形态，不是动态伸缩训练。

权重身份：`policy/snapshot_iteration43100.pt`，迭代 43100，SHA-256 `57f026fdfffeaa7579d9fed8a3ebdd16df3b9f99215038dd02d248c972c09f68`。

## English

```bash
git clone git@github.com:wangshuyiHIT/scaler_mimic_demo.git && cd scaler_mimic_demo && bash demo.sh
```

This opens the looping viewer for iteration 43100, score `[34, 123, 39, 136]` on the fixed 160-grid TRAIN set. Left and right change the clip. Hold up or down to change height. Esc closes the window.
