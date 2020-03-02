# DogsVSCats
## introduction

这个项目是17年kaggle上一个经典的猫狗识别的题目。

最近看了几个pytorch项目的源码，正好用这个项目练手搭建一个相对完整的pytorch的项目进行联习.

采用了pytorch的一个高级抽象库ignite来封装trainer，visdom来实现训练loss和验证集acc的可视化。

## data
*dataset.py*文件里封装了一个`DataSet`类，并根据mode参数可以实现`train`，`evaluate`，`test`三种`DataSet`

`trainSet`里面对图片先`T.Resize(256)`,然后`T.RandomCrop(224)`,增加数据丰富性

`testSet`需要按id排序，最后batch的label是图片id，方便最后生成提交文件

## engine
包括*inference.py*和*trainer.py*两个文件
### inference.py
从`checkpoints/`中提取`model`，然后用`testloader`进行训练，最后将结果拼接成提交的格式，保存为`.csv`文件

每个iteration训练的结果用`TestResult`(继承了`ignite.metrics.Metric`类）拼接在一起

考虑到kaggle用的是log loss作为评分标准的，将预测结果截断到(0.005, 0.995)，最后结果提升了0.02左右。[参考](https://zhuanlan.zhihu.com/p/62317034?utm_source=wechat_sessio)

### trainer.py
先编写了`trainer`和`evaluator`两个`ignite.engine.Engine`，然后通过@修饰器以及`Engine`的`Events`实现在train以及evaluate过程中，各种功能的完成

loss每10个iteration求一次平均值，evaluate每100个iteration计算一次，并用visdom.line(X,Y,update='append')可视化

model的resume是用`ignite.handler.Checkpoint`实现的，这里一直卡了很久，因为到后面验证集精度提高号会出现score相同的情况，导致要保存的文件名重复报错

使用ignite自带的`global_step_from_engine`函数，只能在名字中加入epoch，而我又不想1个epoch保存一次模型，所以还是会面临重复的问题。后面跟据这个函数的源码，拷贝过来，自己改成了返回iteration解决。最后生成的文件名称为iteration+scorename+score，避免了文件名重复的问题

loss默认值为0.001，分别用0.01和0.001训练结果，0.001效果更好，且收敛更快。到达5个epoch后，每两个epoch，lr*0.5缩小

## models/
*ResNet.py*主要还是自己看了别人源码之后，自己再参考写一遍熟悉熟悉，不过开始用这个没有预训练参数的模型的训练效果不理想，还是得使用`torchvision.model`里的预训练过的resnet模型，这里用的是resnet50

AlexNet的实现后面再补充吧

## tmp/
包含了`test/`和`train/`两个文件夹，分别放测试集和训练集

## utils/
写了两个继承`ignite.metrics.Metric`的类，一个用于evaluate的时候计算acc，一个就是前面说的生成test结果

## Last
*config.py*就是保存一些默认参数，比如：保存路径、lr、batchsize等，这样改起来方便。

*main.py*就是直接调用`trainer`开始模型训练了
