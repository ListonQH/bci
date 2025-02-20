# overview
看到一些工作，把小波变换，快速傅里叶变换，融合到网络中。注意，不是用这些方法进行预处理，而是融入到预处理中。

让我看看怎么回事。

MSGNet：https://github.com/YoZhibo/MSGNet。 加了FFT。

小波变换+UNet：https://github.com/thqiu0419/MLWNet。

参考：
1. https://blog.csdn.net/weixin_42645636/article/details/144515576
2. https://mbd.baidu.com/newspage/data/dtlandingsuper?nid=dt_5042578600403937316&sourceFrom=search_a

## Hoooooooo

假设采集1s的脑电信号。其中有效信号占比随着时间的推移而降低。基于这个特性：
1. 用网络模拟变化过程
2. 提高数据中信息的利用率

在图像色彩重建领域，从先验知识中选一个与待重建样本相近的样本，根据这个相似样本对未知样本重建。换句话说，拟合不了变化值，那就观察变化率。求导，求导，继续求导，一直导下去！
