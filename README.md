SAM的地址是这个：[https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)，如何安装以及环境所需要的包都在它的README里面

把main.py放在SAM的工程文件的根目录里，下载好模型参数，然后配置好环境就可以用了
```shell
python main.py --source-video '源视频路径' --target-root '保存根路径' --device 'cuda:0' --model '模型下载的路径'
```
