# 系统演示录制指南

## 推荐工具

macOS 自带即可：

1. 打开 QuickTime Player。
2. 选择“文件 > 新建屏幕录制”。
3. 麦克风选择 MacBook 麦克风或耳机麦克风。
4. 只录终端所在屏幕区域，避免桌面杂乱。

也可以用 OBS，设置为 1920x1080 或 1280x720，帧率 30 fps。

## 录制窗口布置

推荐只放两个窗口：

- 左侧或全屏：终端，字号调到 18-22。
- 右侧可选：Finder/VS Code 文件树，显示 `data/`、`results/`、`report/`、`presentation/`。

如果只录终端，也完全可以。命令输出已经足够证明系统状态。

## 拍摄前检查

```bash
cd /Users/lichenghuang/workspace/homeworks/data-mining/2026-5-30midterm/sentiment-cross-domain
PY=/Users/lichenghuang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
$PY scripts/demo_snapshot.py
$PY scripts/verify_final.py
```

确认两条脚本都能正常输出后再开始录制。

## 正式录制节奏

- 0:00 前：终端里先清屏，输入 `clear`。
- 0:00-0:20：一句话介绍系统。
- 0:20-0:55：展示目录结构。
- 0:55-1:35：运行 `demo_snapshot.py`。
- 1:35-2:10：运行 `verify_final.py`。
- 2:10-2:40：展示 summary 和错误案例。
- 2:40-3:00：展示报告和 PPT 产物。

## 录制时不要做的事

- 不要现场跑完整 `./scripts/run_final.sh`，它会重跑 E0-E6，时间可能超过 1 分钟，视频节奏不好控制。
- 不要临时解释太多代码细节，演示目标是证明系统可运行和结果可追溯。
- 不要把终端字号设太小，老师看不清输出会扣观感分。

## 结尾话术

“以上就是系统演示。可以看到项目从数据文件、实验矩阵、验证脚本到最终报告和答辩材料都已经闭环，所有关键结论都有本地结果文件支撑。”
