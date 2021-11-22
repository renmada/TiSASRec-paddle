# 模型名称 TiSASRec: Time Interval Aware Self-Attention for Sequential Recommendation
## 1. 简介
但是大多数序列化推荐模型都有一个简化的假设，即这些模型都将交互历史视为一个有顺序的序列，没有考虑这个序列中交互物品之间的时间间隔（即只是建模了时间顺序没有考虑实际上的时间戳）。
本论文提出的方法TiSASRec (Time Interval Aware Self-Attention for Sequential Recommendation), 不仅考虑物品的绝对位置,还考虑序列中物品之间的时间间隔
## 2. 复现精度
- 目标精度 (NDCG@10：0.5706，Hit@10：0.8038)
- 复现 (NDCG@10: 0.5712, HR@10: 0.8055)
## 3. 数据集
ml-1m
## 4. 环境依赖
paddlepaddle-gpu=2.2.0
## 6. 训练评估
```
nohup python main.py --dataset=ml-1m --train_dir=default --seed 6 > train.log &
```
### 部分日志
```
epoch:20, time: 60.191668(s), valid (NDCG@10: 0.5411, HR@10: 0.7879), test (NDCG@10: 0.5101, HR@10: 0.7588)
epoch:40, time: 121.114241(s), valid (NDCG@10: 0.5792, HR@10: 0.8156), test (NDCG@10: 0.5540, HR@10: 0.7894)
epoch:60, time: 181.458967(s), valid (NDCG@10: 0.5905, HR@10: 0.8233), test (NDCG@10: 0.5627, HR@10: 0.7977)
epoch:80, time: 241.794130(s), valid (NDCG@10: 0.5922, HR@10: 0.8204), test (NDCG@10: 0.5686, HR@10: 0.7992)
epoch:100, time: 303.143580(s), valid (NDCG@10: 0.5943, HR@10: 0.8210), test (NDCG@10: 0.5712, HR@10: 0.8055)
epoch:120, time: 364.157205(s), valid (NDCG@10: 0.5970, HR@10: 0.8224), test (NDCG@10: 0.5723, HR@10: 0.8005)
```
```
@inproceedings{li2020time,
  title={Time Interval Aware Self-Attention for Sequential Recommendation},
  author={Li, Jiacheng and Wang, Yujie and McAuley, Julian},
  booktitle={Proceedings of the 13th International Conference on Web Search and Data Mining},
  pages={322--330},
  year={2020}
}
```
