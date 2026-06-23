Financial Prediction System

项目简介

本项目为《深度学习与边缘智能》课程期末设计项目。

项目以金融时间序列预测为研究对象，采用循环神经网络（RNN）、门控循环单元（GRU）以及长短期记忆网络（LSTM）三种深度学习模型，对股票及数字货币价格进行预测，并从预测精度、训练效率、模型参数规模以及推理延迟等方面进行比较分析。

通过实验验证不同循环神经网络结构在金融预测任务中的性能差异，并结合边缘计算场景分析模型部署方案。

项目目标

实现RNN时间序列预测模型

实现GRU时间序列预测模型

实现LSTM时间序列预测模型

比较三种模型预测性能

分析边缘设备部署可行性

搭建Web可视化预测系统

项目结构

Financial-Prediction-System

├── data

│ └── btc.csv

├── models

│ ├── rnn.py

│ ├── gru.py

│ └── lstm.py

├── weights

│ ├── rnn.pth

│ ├── gru.pth

│ └── lstm.pth

├── results

│ ├── loss_curve.png

│ ├── prediction.png

│ └── latency.png

├── static

├── templates

│ ├── index.html

│ └── result.html

├── train.py

├── predict.py

├── app.py

├── requirements.txt

└── README.md

数据集

实验采用公开金融市场数据：

比特币价格数据

数据来源：Yahoo Finance

时间范围：2020—2025

数据类型：BTC-USD历史行情

主要特征：

Open

High

Low

Close

Volume

实验环境

硬件环境：

CPU：Intel Core i7

GPU：NVIDIA RTX 4060

RAM：16GB

软件环境：

Python 3.11

PyTorch 2.0

NumPy

Pandas

Scikit-Learn

Flask

模型结构

RNN

输入层 → RNN层 → 全连接层 → 输出层

GRU

输入层 → GRU层 → 全连接层 → 输出层

LSTM

输入层 → LSTM层 → 全连接层 → 输出层

实验结果

ModelRMSERNN0.076GRU0.062LSTM0.056 

实验结果表明：

LSTM预测精度最高

GRU综合性能最佳

RNN训练速度最快

Web系统功能

系统基于Flask框架实现。

主要功能：

上传金融数据

模型选择

实时预测

预测结果可视化

运行方式

安装依赖：

pip install -r requirements.txt

启动系统：

python app.py

浏览器访问：

http://127.0.0.1:5000

作者

课程项目：金融时间序列预测系统

技术路线：

RNN + GRU + LSTM + Flask + PyTorch

License

For Educational Use Only.
