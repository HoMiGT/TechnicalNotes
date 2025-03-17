# Torch 不同层适合的场景
| 任务类型 | 适合的层 |
| :-- | :-- | 
| 普通分类 / 回归 | torch.nn.Linear() | 
| 图像处理 | torch.nn.Conv2d() | 
| 时间序列、语音、文本 | torch.nn.LSTM()、torch.nn.GRU() | 
| 大数据集优化 | torch.nn.BatchNorm1d() | 
