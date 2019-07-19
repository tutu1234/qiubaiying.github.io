---
layout:     post
title:      python web部署
subtitle:   deploy python web service
date:       2019-07-16
author:     Hututu
header-img: img/post-bg-2015.jpg
catalog: true
tags:
    - python web 
---
## python web服务部署

web的部署方面，大都是采用nginx做前端代理，后台采用flask框架，gunicorn作为wsgi容器，同时采用supervisor管理服务器进程，也就是最终的部署方式为：
```
nginx + gunicorn + flask + supervisor
```

#### 创建项目目录
```
cd /tmp
mkdir zhy-image-analysis
```
#### 创建python虚拟环境
使用virtualenv在一个系统中创建不同的python隔离环境，相互之间不会影响。
```
cd zhy-image-analysis
virtualenv img-env   #创建python虚拟环境
source img-env/bin/activate  #激活python虚拟环境
```

#### 安装python web框架flask
1. flask是一个python web framework，简洁高效，使用简单，采用 pip 方式安装即可：
```
pip install flask
```
2. 测试我们的flask安装是否成功，并使用 flask 写一个简单的 web 服务，`vim myapp.py`:
```
from flask import Flask
app = Flask(__name__)
@app.route('/')
def index():
    return 'hello world'
if __name__ == '__main__':
    app.debug = True
    app.run()
```
3. 启动 flask, 用浏览器访问`http://127.0.0.1:5000`就能看到网页显示`hello world`
```
python myapp.py
```
![](http://ww1.sinaimg.cn/large/8833244fly1g51hiupon1j209g049web.jpg)

#### 使用gunicorn部署python web
现在已经使用flask自带的服务器，完成了web服务的启动。生产环境下，flask 自带的服务器，无法满足性能要求。我们这里采用gunicorn做wsgi容器，用来部署python web服务
1. 安装 gunicorn
```
 pip install gunicorn
```
2. 使用`pip freeze`,每次使用 pip安装的库，都写入一个requirement文件里面，既能知道自己安装了什么库，也方便别人部署时，安装相应的库。
```
 pip freeze > requirements.txt
```
以后每次 pip 安装了新的库的时候，都需freeze 一次。
3. 安装好gunicorn之后，用gunicorn启动flask，上面的代码启动了`app.run()`,是用flask自带的服务器启动app。这里我们使用了gunicorn，此时myapp.py就等同于一个库文件，被gunicorn调用。
```
 gunicorn -w4 -b0.0.0.0:8000 myapp:app
```
这里，我们用`8000`的端口进行访问，原先的`5000`并没有启用。其中gunicorn 的部署中，`-w`表示开启多少个`worker`，`-b`表示gunicorn的访问地址。
![](http://ww1.sinaimg.cn/large/8833244fly1g51ij1e0rej20j203hjrj.jpg)想要结束gunicorn需执行`pkill gunicorn`，有时候还的`ps`找到`pid`进程号才能`kill`。可是这对于一个开发来说，太过于繁琐，因此出现了另外一个神器`supervisor`，一个专门用来管理进程的工具，还可以管理系统的工具进程。

#### 使用supervisor管理进程
1. 安装supervisor
```
pip install supervisor
echo_supervisord_conf > supervisor.conf   # 生成 supervisor 默认配置文件
vim supervisor.conf                       # 修改 supervisor 配置文件，添加 gunicorn 进程
```
2. 在`zhy-image-analysis/supervisor.conf`配置文件底部添加
```
[include]
files=/tmp/zhy-image-analysis/myapp.conf
```
`/tmp/zhy-image-analysis/myapp.conf`的内容为:
```
[program:myapp]
command=/tmp/zhy-image-analysis/img-env/bin/gunicorn -w4 -b0.0.0.0:2170 myapp:app #supervisor启动命令
directory=/tmp/zhy-image-analysis                                            #项目的文件夹路径
startsecs=0                                                                  #启动时间
stopwaitsecs=0                                                               #终止等待时间
autostart=false                                                              #是否自动启动
autorestart=false                                                            #是否自动重启
stdout_logfile=/tmp/zhy-image-analysis/log/gunicorn.log                      #log 日志
stderr_logfile=/tmp/zhy-image-analysis/log/gunicorn.err
```
supervisor的基本使用命令
```
supervisord -c supervisor.conf                           通过配置文件启动supervisor
supervisorctl -c supervisor.conf start myapp             启动指定/所有 supervisor管理的程序进程
supervisorctl -c supervisor.conf status                  察看supervisor的状态
supervisorctl -c supervisor.conf reload                  重新载入 配置文件
supervisorctl -c supervisor.conf stop  myapp             关闭指定/所有 supervisor管理的程序进程
```
访问`http://127.0.0.1:2170`可以看见`gunciron`启动的返回的`hello world`

