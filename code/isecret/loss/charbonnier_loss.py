import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (pseudo Huber Loss) implementation."""
    def __init__(self, epsilon=1e-6):
        """
        初始化Charbonnier Loss函数
        参数:
        - epsilon: 平滑项，避免除以零的错误，默认值1e-6
        """
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target):
        """
        计算损失值
        参数:
        - prediction: 预测值
        - target: 真实值
        返回:
        - loss: 计算出的Charbonnier Loss值
        """
        diff = prediction - target
        loss = torch.sqrt(diff * diff + self.epsilon ** 2).mean()  # 按所有元素计算平均损失
        return loss
