SAM的地址是这个：[https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)，如何安装以及环境所需要的包都在它的README里面

把main.py放在SAM的工程文件的根目录里，下载好模型参数（放在model目录下面），然后配置好环境就可以用了

shell语句如下：
```shell
python main.py --source-video '源视频路径' --target-root '保存根路径' --device 'cuda:0' --model '模型下载的路径'
```
例如：
```shell
python main.py --source-video './assets/cxk.mp4' --target-root './assets' --device 'cuda:0' --model './model/sam_vit_h_4b8939.pth'
```
