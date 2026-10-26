"""为什么 no-bn 那组停在 10%？—— 64 张随机图跑 40 步 SGD，看网络到底能不能拟合。

    python nobn_probe.py

一个过参数化的网络应该能瞬间记住 64 张随机图。开着 BN 确实做到了；关掉之后信号要么逐层消失
（xavier），要么直接爆炸（kaiming），所以 no-bn 那组不管训多久都停在随机猜。
"""
import torch

from cifarlab.models.googlenet import GoogLeNet

torch.manual_seed(0)
X, y = torch.randn(64, 3, 32, 32), torch.randint(0, 10, (64,))
loss = torch.nn.CrossEntropyLoss()
print(f'{"setting":34s} {"loss@1":>8} {"loss@40":>9} {"logit std":>10}')
for bn, init, lr in [(True, None, 0.05), (False, None, 0.05), (False, 'xavier', 0.05),
                     (False, 'kaiming', 0.05)]:
    torch.manual_seed(0)
    net = GoogLeNet(bn=bn, init=init)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    first = None
    for _ in range(40):
        opt.zero_grad()
        l = loss(net(X), y)
        l.backward()
        opt.step()
        first = l.item() if first is None else first
    print(f'bn={str(bn):5s} init={str(init):7s} lr={lr:<4} '
          f'{first:8.3f} {l.item():9.3f} {net(X).std().item():10.3f}')
