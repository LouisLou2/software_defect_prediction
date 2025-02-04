# 定义 SupConLoss
import torch
from torch import nn
from torch.nn import functional as F

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, label):
        device = features.device
        n = features.shape[0]  # batch
        T = self.temperature  # 温度参数T
        # label=label.squeeze() no need
        # 这步得到它的余弦相似度矩阵
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        # 这步得到它的label矩阵，相同label的位置为1
        mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))

        # 这步得到它的不同类的矩阵，不同类的位置为1
        mask_no_sim = torch.ones_like(mask) - mask

        # 这步产生一个对角线全为0的，其他位置为1的矩阵
        mask_dui_jiao_0 = (torch.ones(n, n) - torch.eye(n, n)).to(device)

        # 这步给相似度矩阵求exp,并且除以温度参数T
        similarity_matrix = (torch.exp(similarity_matrix / T)).to(device)

        # 这步将相似度矩阵的对角线上的值全置0，因为对比损失不需要自己与自己的相似度
        similarity_matrix = similarity_matrix * mask_dui_jiao_0

        # 这步产生了相同类别的相似度矩阵，标签相同的位置保存它们的相似度，其他位置都是0，对角线上也为0
        sim = mask * similarity_matrix

        # 用原先的对角线为0的相似度矩阵减去相同类别的相似度矩阵就是不同类别的相似度矩阵
        no_sim = similarity_matrix - sim

        # 把不同类别的相似度矩阵按行求和，得到的是对比损失的分母(还差一个与分子相同的那个相似度，后面会加上)
        no_sim_sum = torch.sum(no_sim, dim=1)

        '''
        将上面的矩阵扩展一下，再转置，加到sim（也就是相同标签的矩阵上），然后再把sim矩阵与sim_num矩阵做除法。
        至于为什么这么做，就是因为对比损失的分母存在一个同类别的相似度，就是分子的数据。做了除法之后，就能得到
        每个标签相同的相似度与它不同标签的相似度的值，它们在一个矩阵（loss矩阵）中。
        '''
        no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
        sim_sum = sim + no_sim_sum_expend
        loss = torch.div(sim, sim_sum)

        '''
        由于loss矩阵中，存在0数值，那么在求-log的时候会出错。这时候，我们就将loss矩阵里面为0的地方
        全部加上1，然后再去求loss矩阵的值，那么-log1 = 0 ，就是我们想要的。
        '''
        loss = mask_no_sim.to(device) + loss.to(device) + torch.eye(n, n).to(device)

        # 接下来就是算一个批次中的loss了
        loss = -torch.log(loss)  # 求-log
        tmp = torch.nansum(loss, dim=1)
        # 判断是否是nan
        if torch.isnan(tmp).sum() > 0:
            print('nan exists')

        # 注意以下两步，如果构建的全是负对或者全是正对，就会出现nan的情况，使用nansum而不是sun, nansum会把当做0看待
        loss = torch.nansum(tmp) / (2 * n)  # 将所有数据都加起来除以2n
        # 判断是否是nan
        if torch.isnan(loss).sum() > 0:
            print('nan exists')
        return loss