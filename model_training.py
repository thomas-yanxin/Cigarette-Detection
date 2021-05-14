import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from paddlex.det import transforms
import paddlex as pdx



# 定义训练和验证时的transforms
# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html
train_transforms = transforms.Compose([
    transforms.MixupImage(mixup_epoch=250), transforms.RandomDistort(),
    transforms.RandomExpand(), transforms.RandomCrop(), transforms.Resize(
        target_size=608, interp='RANDOM'), transforms.RandomHorizontalFlip(),
    transforms.Normalize()
])

eval_transforms = transforms.Compose([
    transforms.Resize(
        target_size=608, interp='CUBIC'), transforms.Normalize()
])

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-vocdetection
train_dataset = pdx.datasets.VOCDetection(
    data_dir='/home/aistudio/dataset',
    file_list='/home/aistudio/dataset/train_list.txt',
    label_list='/home/aistudio/dataset/labels.txt',
    transforms=train_transforms,
    parallel_method='thread',
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='/home/aistudio/dataset',
    file_list='/home/aistudio/dataset/val_list.txt',
    label_list='/home/aistudio/dataset/labels.txt',
    parallel_method='thread',
    transforms=eval_transforms)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://paddlex.readthedocs.io/zh_CN/develop/train/visualdl.html
num_classes = len(train_dataset.labels)

# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-ppyolo
model = pdx.det.PPYOLO(num_classes=num_classes)

# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#train
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=540,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    save_interval_epochs=1,
    lr_decay_epochs=[270,320, 480],
    save_dir='output/ppyolo',
    use_vdl=True)