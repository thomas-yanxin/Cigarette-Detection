
[AIStudio地址](https://aistudio.baidu.com/aistudio/projectdetail/1943814)
<font face="黑体" size=4>不抽烟，也没有烟，所以只能以这种形式代替了</font>


![](https://ai-studio-static-online.cdn.bcebos.com/56d5a3ec5da6480ba188da3294d47d0549c117988a8a4aa39b071ea0ec731ab8)


# 爸，这下你还敢抽烟吗？/邪笑
![](https://ai-studio-static-online.cdn.bcebos.com/b6813e004b8549bfb791d6cbafebdc77f0319d85c1d34ce5ba0a57e6098cde60)  

<font face="黑体" size=4>我爸：这锅我不背！！</font>

## 项目背景
<font face="宋体" size=4>
&emsp;&emsp;2014年11月24日，卫生计生委起草了《公共场所控制吸烟条例（送审稿）》向社会公开征求意见。送审稿明确，所有室内公共场所一律禁止吸烟。此外，体育、健身场馆的室外观众坐席、赛场区域；公共交通工具的室外等候区域等也全面禁止吸烟。而《公共场所控制吸烟条例》实施后，卫计委将考虑为控烟条例设置举报电话，同时开展监测评估，鼓励全社会参与戒烟活动。  

&emsp;&emsp;但有很多人秉持“多一事不如少一事”的原则：“**欸，有点电话就不打，我就忍着！**”，导致社会公共场所禁烟效果普遍不佳。  

&emsp;&emsp;因而“公共场所，禁止吸烟”的标牌，也渐渐沦为了摆设。   
  
  
![](https://ai-studio-static-online.cdn.bcebos.com/60808c2936c248318eed57450e5c7f1046349db32f7a4983946ed9ed02af1b78)

  
  </font>

## 项目介绍 

<font face="宋体" size=4>
&emsp;&emsp;本项目基于PaddleX进行模型训练，并利用inference导出后部署在本地，并使用百度大脑合成后进行“公共场所，禁止吸烟”的语音播报。
  
  </font>

### 首先导入使用PaddleX时所需的模块


```python
!pip install paddlex
!pip install paddle2onnx
```


```python
# 进行数据集解压
!unzip -oq /home/aistudio/data/data86368/smoke.zip -d /home/aistudio/dataset2
```

<font face="宋体" size=4>

&emsp;&emsp;下面这个check.py是我在开发过程中遇到点小问题在PaddleX的Github issue中找到的，后面我们会提到这个脚本的功能。  
**（深刻启示：有问题就去逛issue！！！要相信你现在所遇到的绝大多数问题已经有“先烈”替你趟过了！即使没有也可以提issue!飞桨的开发小哥哥们回复也是炒鸡快！！！！）**
  
  


```python
!mv check.py dataset
```

### 数据处理和数据清洗

<font face="宋体" size=4>
&emsp;&emsp;在我们解压好数据集后，我们可以看到数据集中给了images和Annotations两个文件夹，分别存放着.jpg和.xml文件。但其实这是不符合PaddleX在目标检测中的数据格式要求的。并且，我们打开他的.xml文件可以发现，它的filename、name、path等内容均需要我们后期进行处理和更改。  

&emsp;&emsp;当然，在本项目中的数据处理方式有些许笨拙和低劣。



```python
# 这里修改.xml文件中的<path>元素
!mkdir dataset/Annotations1
import xml.dom.minidom
import os

path = r'dataset/Annotations'  # xml文件存放路径
sv_path = r'dataset/Annotations1'  # 修改后的xml文件存放路径
files = os.listdir(path)
cnt = 1

for xmlFile in files:
    dom = xml.dom.minidom.parse(os.path.join(path, xmlFile))  # 打开xml文件，送到dom解析
    root = dom.documentElement  # 得到文档元素对象
    item = root.getElementsByTagName('path')  # 获取path这一node名字及相关属性值
    for i in item:
        i.firstChild.data = '/home/aistudio/dataset/JPEGImages/' + str(cnt).zfill(6) + '.jpg'  # xml文件对应的图片路径

    with open(os.path.join(sv_path, xmlFile), 'w') as fh:
        dom.writexml(fh)
    cnt += 1
```


```python
# 这里修改.xml文件中的<failname>元素
!mkdir dataset/Annotations2
import xml.dom.minidom
import os

path = r'dataset/Annotations1'  # xml文件存放路径
sv_path = r'dataset/Annotations2'  # 修改后的xml文件存放路径
files = os.listdir(path)

for xmlFile in files:
    dom = xml.dom.minidom.parse(os.path.join(path, xmlFile))  # 打开xml文件，送到dom解析
    root = dom.documentElement  # 得到文档元素对象
    names = root.getElementsByTagName('filename')
    a, b = os.path.splitext(xmlFile)  # 分离出文件名a
    for n in names:
        n.firstChild.data = a + '.jpg'
    with open(os.path.join(sv_path, xmlFile), 'w') as fh:
        dom.writexml(fh)
```


```python
# 这里修改.xml文件中的<name>元素
!mkdir dataset/Annotations3
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
 
import os
import xml.etree.ElementTree as ET
 
origin_ann_dir = '/home/aistudio/dataset/Annotations2/'# 设置原始标签路径为 Annos
new_ann_dir = '/home/aistudio/dataset/Annotations3/'# 设置新标签路径 Annotations
for dirpaths, dirnames, filenames in os.walk(origin_ann_dir):   # os.walk游走遍历目录名
  for filename in filenames:
    print("process...")
    # if os.path.isfile(r'%s%s' %(origin_ann_dir, filename)):   # 获取原始xml文件绝对路径，isfile()检测是否为文件 isdir检测是否为目录
    origin_ann_path = os.path.join(origin_ann_dir, filename)   # 如果是，获取绝对路径（重复代码）
    new_ann_path = os.path.join(new_ann_dir, filename)
    tree = ET.parse(origin_ann_path)  # ET是一个xml文件解析库，ET.parse（）打开xml文件。parse--"解析"
    root = tree.getroot()   # 获取根节点
    for object in root.findall('object'):   # 找到根节点下所有“object”节点
        name = str(object.find('name').text)  # 找到object节点下name子节点的值（字符串）
        # 如果name等于str，则删除该节点
        if (name in ["smoke"]):
        #   root.remove(object)
            pass

        # 如果name等于str，则修改name
        else:
            object.find('name').text = "smoke"

        tree.write(new_ann_path)#tree为文件，write写入新的文件中。
        print("OK1")
```


<font face="宋体" size=4>
由于在上述数据处理过程中我们产生了很多冗余的文件，故需将其删除。并将其更改为适合PaddleX的数据集格式。


```python
!rm -rf dataset/Annotations
!rm -rf dataset/Annotations1
!rm -rf dataset/Annotations2
!mv dataset/Annotations3 dataset/Annotations
!mv dataset/images dataset/JPEGImages
```

在原始数据集中，存在.jpg文件和.xml文件匹配不对等的情况，这里我们根据.jpg文件名删除了在Annotations文件夹中无法匹配的.xml文件，使得.jpg和.xml能够一一对应。


```python
import os
import shutil
path_annotations = 'dataset/Annotations'
path_JPEGImage = 'dataset/JPEGImages'
xml_path = os.listdir(path_annotations)
jpg_path = os.listdir(path_JPEGImage)
for i in jpg_path:
    a = i.split('.')[0] + '.xml'
    if a in xml_path:
        pass
    else:
        print(i)
        os.remove(os.path.join(path_JPEGImage,i))
```

<font face="宋体" size=4>

因为懒(有现成的为啥还要自己写呢~)，所以就用了PaddleX 自带的划分数据集的命令，这里我们训练集、验证集、测试集的比例为7:2:1。


```python
!paddlex --split_dataset --format VOC --dataset_dir /home/aistudio/dataset/ --val_value 0.2 --test_value 0.1
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/setuptools/depends.py:2: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    Dataset Split Done.[0m
    [0mTrain samples: 3403[0m
    [0mEval samples: 972[0m
    [0mTest samples: 486[0m
    [0mSplit files saved in /home/aistudio/dataset/[0m
    [0m[0m

<font face="宋体" size=4>

&emsp;&emsp;当然，从一般情况来说，这一步完成之后就可以进行模型训练了。但其实在实际过程中，若进行模型训练，在训练过程中会报错。具体报错，大家感兴趣的可以实地体验一下（嘿嘿嘿）。这里就需要用到我们在开头所提到的check.py文件。执行以下命令，会输出图片读取有误的图片路径。我们用rm -rf命令将其与其对应的.xml文件删除。


```python
%cd dataset/
!cat train_list.txt | python check.py
!cat test_list.txt | python check.py
!cat val_list.txt | python check.py
```

    /home/aistudio/dataset
    Wrong img file: JPEGImages/790_strawberry.jpg
    Wrong img file: JPEGImages/791_strawberry.jpg
    Wrong img file: JPEGImages/000143.jpg
    Wrong img file: JPEGImages/000144.jpg
    Wrong img file: JPEGImages/000120.jpg
    Wrong img file: JPEGImages/780_strawberry.jpg
    Wrong img file: JPEGImages/845_strawberry.jpg
    Wrong img file: JPEGImages/smoke132.jpg
    Corrupt JPEG data: 1 extraneous bytes before marker 0xdb
    Wrong img file: JPEGImages/000325.jpg
    Wrong img file: JPEGImages/000193.jpg



```python
!rm -rf JPEGImages/000120.jpg
!rm -rf JPEGImages/000143.jpg
!rm -rf JPEGImages/791_strawberry.jpg
!rm -rf JPEGImages/smoke132.jpg
!rm -rf JPEGImages/780_strawberry.jpg
!rm -rf JPEGImages/000193.jpg
!rm -rf JPEGImages/845_strawberry.jpg
!rm -rf JPEGImages/000325.jpg
!rm -rf JPEGImages/000144.jpg
!rm -rf JPEGImages/790_strawberry.jpg
!rm -rf Annotations/000120.xml
!rm -rf Annotations/000143.xml
!rm -rf Annotations/791_strawberry.xml
!rm -rf Annotations/smoke132.xml
!rm -rf Annotations/780_strawberry.xml
!rm -rf Annotations/000193.xml
!rm -rf Annotations/845_strawberry.xml
!rm -rf Annotations/000325.xml
!rm -rf Annotations/000144.xml
!rm -rf Annotations/790_strawberry.xml
```

<font face="宋体" size=4>

并且删除之前生成的几个关于数据集的文本文件，重新执行PaddleX划分数据集的命令。


```python
!rm -rf val_list.txt
!rm -rf train_list.txt
!rm -rf test_list.txt
!rm -rf labels.txt
```


```python
!paddlex --split_dataset --format VOC --dataset_dir /home/aistudio/dataset/ --val_value 0.2 --test_value 0.1
```


```python
%cd /home/aistudio/
```

    /home/aistudio


## 模型训练

<font face="宋体" size=4>

&emsp;&emsp;在使用PaddleX进行模型训练的过程中，我们使用目前PaddleX适配精度最高的PPYolo模型进行训练。其模型较大，预测速度比YOLOv3-DarkNet53更快，适用于服务端。大家也可以更改其他模型尝试一下。这里我训练了大概300个epoch(别问，问就是没算力了也懒得续点了……)其中第277个epoch效果最佳，map大概在76。（有算力的童鞋可以试着调参或者继续往下面去试试）


```python
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
```

## 模型导出

<font face="宋体" size=4>

&emsp;&emsp;这里我们将训练过程中保存的模型导出为inference格式模型，其原因在于：PaddlePaddle框架保存的权重文件分为两种：支持前向推理和反向梯度的训练模型 和 只支持前向推理的推理模型。二者的区别是推理模型针对推理速度和显存做了优化，裁剪了一些只在训练过程中才需要的tensor，降低显存占用，并进行了一些类似层融合，kernel选择的速度优化。而导出的inference格式模型包括__model__、__params__和model.yml三个文件，分别表示模型的网络结构、模型权重和模型的配置文件（包括数据预处理参数等）。


```python
!paddlex --export_inference --model_dir=PaddleDetection/output/ppyolo/epoch_277 --save_dir=./inference_model
```

### 简单部署尝试

<font face="宋体" size=4>

&emsp;&emsp;下面我们简单地部署体验了一番。这里打开了设备摄像头进行视频流预测，并设置了在阈值>=0.3的时候会播放“公共场所，禁止吸烟！”  

&emsp;&emsp;在本项目中可以看到存在“cigarette.mp3”和“dad.mp3”的音频文件。“dad.mp3”的具体内容这里卖个关子。大家下载下来体验一下就知道了hhh。  

&emsp;&emsp;而这里的音频，使用的是百度大脑的语音合成API，具体的大家可以去百度大脑官网看看，在本项目中就不予展示了。


```python
import cv2
import paddlex as pdx
from playsound import playsound

# 修改模型所在位置
predictor = pdx.deploy.Predictor('D:\\project\\python\\cigarette\\inference_model')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        result = predictor.predict(frame)
        score = result[0]['score']
        if score >= 0.3:
            print("*"*100)
            # 修改音频所在位置
            playsound('D:\\project\\python\\cigarette\\cigarette.mp3')
        # print(result)
        vis_img = pdx.det.visualize(frame, result, threshold=0.6, save_dir=None)
        cv2.imshow('cigarette', vis_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
```

### 个人相关介绍
- 笔名：左右
- 华东理工大学自动化专业大二在读
- 号称：冷板凳常客

- 热衷于利用人工智能技术做点有价值的东西，为社会做点小事情；
- 另外也研究点多智能体的东西；
- 偶尔写点随笔、摄影、仰望星空.....

[AiStudio主页，欢迎互关哟](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/383005)  

[Github主页，欢迎互关哟](https://github.com/thomas-yanxin)  

[CSDN主页，欢迎互关哟](https://blog.csdn.net/Mefishes?spm=1000.2115.3001.5343)  

请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
