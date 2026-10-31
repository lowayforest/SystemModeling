# 系统辨识作业代码

本仓库按题目保存仿真代码。当前包含第一题的 4 阶 M 序列生成和第二题的最小二乘一次完成算法。教材依据为《系统辨识》印刷第 55 页例 2.6 和第 138 页例 5.3。

## 配置 Conda 环境

首次使用时运行一次：

```bash
conda create -n systemmodeling python=3.11 numpy
conda activate systemmodeling
```

如果环境已经创建，跳过第一行。以后打开新终端，只需运行 `conda activate systemmodeling`。如果终端提示需要初始化 Conda，先运行 `conda init zsh`，关闭并重新打开终端，再激活环境。激活成功后，终端提示符前会出现 `(systemmodeling)`。

## 第一题：M 序列生成

运行：

```bash
python m_sequence.py
```

`m_sequence.py` 从初始状态 `1010` 出发，取最右位为输出，使用 `C3 XOR C4` 反馈。它检查序列周期为 15，并把二进制位映射为 `-1/+1`，生成前 400 个输入值。结果写入 `output/m_sequence_400.csv`。M 序列是确定性的，不使用随机种子。

## 第二题：最小二乘一次完成算法

激活上述 Conda 环境后，在本目录运行：

```bash
python least_squares_batch.py
```

`least_squares_batch.py` 调用第一题生成的输入，按教材例 5.3 生成 400 个输出值，再构造数据矩阵并一次求出四个参数。仿真采用 `z(0)=z(-1)=u(0)=u(-1)=0`、噪声随机种子 `26721988`；运行结果写入 `output/least_squares_data_400.csv`。一次运行得到的估计值约为 `(-1.513049, 0.720649, 1.000964, 0.488937)`。

教材没有给出生成表 5.1 时使用的具体噪声样本，因此本仓库的估计值不会与表中数字完全相同。`output/*.csv` 是运行时生成的数据，已加入 `.gitignore`；需要时重新运行程序即可。
