# Flexiv踩坑经验

## 基本构成
Flexiv主要分为机械臂、控制箱(control box)、手柄(motion bar)、平板（UI tablet）几部分。我们一般用网线将电脑、平板、手柄连接至控制箱。平板上有可视化的软件可以读取/更改机器人的状态。利用手柄与平板还可以进入Freedrive模式从而用手调整机械臂的位置。

## 启动Flexiv
确保急停按钮处于按下状态，打开控制箱的开关。等待至机械臂上的环状LED灯亮起，正常情况应呈现蓝色。此时手柄周围的LED灯应该呈红色，转动释放红色的急停按钮，按钮弹起后手柄LED灯光为淡蓝色。可以打开平板上的UI App，观察机器人的状态。至此，Flexiv已成功启动。

## 关闭Flexiv
若处于Freedrive模式最好先退出该模式（具体操作会在下面的模式切换部分叙述）。确保机械臂位置正常后按下红色急停按钮，应该会听见咔擦一声。之后将控制箱开关关闭即可。

## 模式切换

### 急停模式
按下手柄上红色的急停按钮后会进入急停模式，将按钮旋转使其释放、弹上来后可以解除。

### Freedrive模式
* Freedrive模式必须在Manual mode下才能进入。手柄上的switch推向离线近的一端时对应的是Manual mode，另一端则是Auto Mode。切换至Auto mode时需要在平板的UI App上确认（Confirm）；切换至Manual mode则不需要。
* 进入Manual mode后，按下手柄上的带有人的头像的按钮，此时手柄周围的LED灯会从淡蓝色变为深蓝色，UI App上会弹出控制机械臂的界面。可以在平板上操作控制机械臂。若希望使用手与机械臂交互，需要先关闭UI App中弹出的界面，接着用手将手柄侧边键按下至一半处并保持住，与此同时尝试用手与机械臂交互即可。

### External模式
Automode下分为External与Internal模式，希望用电脑控制机械臂的话必须切换至External模式。将手柄上的switch推至远离线的一侧，UI App上会弹出界面，选择External后点Confirm即可。

## 连接电脑
将电脑使用网线连接至控制箱，可以ping机械臂的ip地址（默认为192.168.2.100）确认连接无误。使用代码控制的时候除了机械臂的ip还需要控制电脑的ip，使用ifconfig命令查看确认即可（实验室专门用来连飞夕的电脑的ip应该是192.168.2.108）。之后就可以使用代码控制机械臂的运动了。

## 常见问题

### 机械臂发生错误
这个比较复杂，如果是比较简单的错误一般回到manual mode后按下急停重启机械臂即可。如果依然有错误，可以尝试进入Freedrive模式修正指正常悬浮状态。

### External模式无法启动
有时切换至Automode的时候，UI App上只有Internal选项，从设置里点进External则会显示"External mode is not enabled"。这里有两种可能，第一种就是Flexiv自己的问题，直接按照正常关闭Flexiv的流程关闭Flexiv，并重启平板的UI App，重启后确认问题是否依然存在，一般尝试两次就好了。

另一种可能是，最近有维护人员来对Flexiv的系统进行了更新，那么os中的配置文件里可能把External Mode关闭了。那么可以进行如下操作：
1. 在连接控制箱的电脑终端中运行`telnet 192.168.2.100`（这是Flexiv的默认ip地址）
2. 输入用户名和密码（均为root）登录
3. 输入`cd mnt/programs/specs/robots`，然后进入名称含有qnx的文件夹
4. 通过vi查看名称含有run bot的sh脚本的内容，里面包含一个xml文件的名称，按4中的指示修改该xml文件（也位于当前文件夹下）的内容即可。
5. 找到External Interface部分，将`<enable> 0 </enable>`修改为`<enable> 1 </enable>`。将`<interafce_config_file>externalIOConfig.xml</interafce_config_file>`修改为`<interafce_config_file>externalEthernetConfig.xml</interafce_config_file>`。
6. 退出后重启即可。
