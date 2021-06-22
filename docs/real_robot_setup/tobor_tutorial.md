# UR5踩坑经验

## 基本构成

## 启动与关闭UR5
* 打开连接UR5的平板的电源键，等待进度条加载完毕。点击Go to initialization screen后，点击on。之后点击start，可以听见关节处的响声，至此UR5已成功启动。状态应该是绿色的圆点，并写着Normal。
* 关闭时在平板的主页上点击Shutdown Robot，选择Power off即可。


## 控制机械臂

### 电脑控制
在连接UR5的电脑上可以使用代码控制机械臂的运动，例如使用[urx](https://github.com/SintefManufacturing/python-urx)（python或c++）库。UR5现在的ip地址为192.168.2.102，可以在平板上查看或修改。

### Freedrive
连接UR5的平板背后有一个黑色按钮，按住它的同时直接用手与机械臂交互即可。松开它则会退出Freedrive模式。

## 更多信息
更多功能以及安全方面的信息可以在[用户手册](https://s3-eu-west-1.amazonaws.com/ur-support-site/22046/UR5_User_Manual_en_Global.pdf)进行查阅。