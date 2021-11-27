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
epoch:60, time: 186.444842(s), valid (NDCG@10: 0.5910, HR@10: 0.8227), test (NDCG@10: 0.5642, HR@10: 0.7985)
epoch:80, time: 248.875761(s), valid (NDCG@10: 0.5929, HR@10: 0.8232), test (NDCG@10: 0.5693, HR@10: 0.8018)
epoch:100, time: 310.499432(s), valid (NDCG@10: 0.5938, HR@10: 0.8210), test (NDCG@10: 0.5738, HR@10: 0.8060)
epoch:120, time: 372.769209(s), valid (NDCG@10: 0.5985, HR@10: 0.8222), test (NDCG@10: 0.5732, HR@10: 0.8015)
epoch:140, time: 434.958108(s), valid (NDCG@10: 0.5990, HR@10: 0.8227), test (NDCG@10: 0.5716, HR@10: 0.8022)
```
## 7. TIPC测试
```
cd PaddleRec
bash test_tipc/prepare.sh ./test_tipc/configs/tisas/train_infer_python.txt 'lite_train_lite_infer'
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/tisas/train_infer_python.txt 'lite_train_lite_infer'
```
## 8. 精度对齐
```
python compare.py
````
```
# 引用原论文
```
@inproceedings{li2020time,
  title={Time Interval Aware Self-Attention for Sequential Recommendation},
  author={Li, Jiacheng and Wang, Yujie and McAuley, Julian},
  booktitle={Proceedings of the 13th International Conference on Web Search and Data Mining},
  pages={322--330},
  year={2020}
}
```
