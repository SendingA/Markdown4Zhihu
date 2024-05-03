

# 毫米微波雷达4D点云成像的处理流程

编写：刘圣鼎

指导：张进、邱彦龙

- 经ADC采样后得到的雷达数据，将其重排成三个维度，距离维/速度维/天线维（sample/chirp/channel)

- 做完Range FFT（记得加窗，比如，切比雪夫，SVA（Spatial Variant Apodization）[自适应加窗]）之后，砍掉负频率（负距离对于雷达没有任何有用的信息），sample维变成range bin，大小为 <img src="https://www.zhihu.com/equation?tex=\frac{sample\ number}{2}" alt="\frac{sample\ number}{2}" class="ee_img tr_noresize" eeimg="1"> 

- 这一步为可选->BWE（ Bandwidth Extrapolation）作用：提高分辨率，比如128个sample，补成256个sample
  - Python library for radar Bandwidth Extrapolation 

- 做2D FFT（记得加窗，比如，切比雪夫，SVA（Spatial Variant Apodization）[自适应加窗]）得到RD MAP
  - 建议把每个channel对应的RD Map压在一起（非相位累加），最后得到一个比较好的RD MAP，压得过程为直接相加，最后除以channel的数量

- 得到RD MAP以后，对于RD MAP做CFAR，估出一个噪底，一般取频谱的中位数，直接用现成的CA-CFAR就行

  - CFAR输入：RD MAP，相当于每个距离每个速度处的强度

  - CFAR输出：把幅值高的亮点的range index和velocity index提取出来

- 对于CFAR出来的每个目标点进行遍历，即把某个range index和velocity index的所有channel提取出来

- 从提取的channel中，把水平的天线提取出来，把竖直的天线提取出来
- 接下来进行水平角和俯仰角的解算，详见下面两个板块
- 最后每个目标点得到4D信息， <img src="https://www.zhihu.com/equation?tex=（r ,v ,\theta ,  \phi , strength）" alt="（r ,v ,\theta ,  \phi , strength）" class="ee_img tr_noresize" eeimg="1"> ，有了4D信息即可进行后续的点云成像以及机器学习或深度学习的算法



## 水平角的结算（以三发四收为例）：

原来我们计算水平角采取的做法为对于8根水平的虚拟天线做FFT，由于天线的数量比较少，采样的点数比较少，因此通常在后面补零，然后做FFT

此处我们计算水平角使用的是导向矢量(Steering Vector)的方法，其实是换了一种方式的FFT

参考：[阵列导向矢量（Steering vector）推导-CSDN博客](https://blog.csdn.net/UncleWa/article/details/123780502)

根据TI手册可将天线阵列的位置计算出来，经计算完全均匀排布，所以实质上是ULA(Unified Line Array)的Steering Vector

![image-20240309231921108](https://raw.githubusercontent.com/SendingA/Markdown4Zhihu/master/Data/毫米微波雷达4D点云成像的处理流程/image-20240309231921108.png)

如图所示，考虑M元均匀线阵(Uniform Linear Array, ULA)，阵元间距 <img src="https://www.zhihu.com/equation?tex=d=\lambda/2" alt="d=\lambda/2" class="ee_img tr_noresize" eeimg="1"> ， <img src="https://www.zhihu.com/equation?tex=D" alt="D" class="ee_img tr_noresize" eeimg="1"> 个波长为 <img src="https://www.zhihu.com/equation?tex=λ" alt="λ" class="ee_img tr_noresize" eeimg="1"> 的远场信号入射到该阵列，入射方向与阵列法线夹角定义为入射角度 <img src="https://www.zhihu.com/equation?tex=θ _i (i= 1,2,...,D )" alt="θ _i (i= 1,2,...,D )" class="ee_img tr_noresize" eeimg="1">  ，则该阵列的导向矩阵可表示为： 

 <img src="https://www.zhihu.com/equation?tex=A(\theta)=[a(\theta_1),a(\theta_2),a(\theta_3),...,a(\theta_D)]" alt="A(\theta)=[a(\theta_1),a(\theta_2),a(\theta_3),...,a(\theta_D)]" class="ee_img tr_noresize" eeimg="1"> 

 <img src="https://www.zhihu.com/equation?tex=A(\theta)" alt="A(\theta)" class="ee_img tr_noresize" eeimg="1"> 是 <img src="https://www.zhihu.com/equation?tex=M\times D" alt="M\times D" class="ee_img tr_noresize" eeimg="1"> 维的导向矩阵，由线性无关的阵列导向矢量 <img src="https://www.zhihu.com/equation?tex=a(\theta_i)" alt="a(\theta_i)" class="ee_img tr_noresize" eeimg="1"> 组成，阵列导向矢量定义为：

 <img src="https://www.zhihu.com/equation?tex=a(\theta_i)=[1,e^{   j2\pi \frac{dsin\theta_i}{\lambda}     } ,...,e^{   j2\pi (M-1)\frac{dsin\theta_i}{\lambda}     }      ]" alt="a(\theta_i)=[1,e^{   j2\pi \frac{dsin\theta_i}{\lambda}     } ,...,e^{   j2\pi (M-1)\frac{dsin\theta_i}{\lambda}     }      ]" class="ee_img tr_noresize" eeimg="1"> 

其中， <img src="https://www.zhihu.com/equation?tex=\theta" alt="\theta" class="ee_img tr_noresize" eeimg="1"> 的精度（D的大小）由我们自行设置，比如 <img src="https://www.zhihu.com/equation?tex=\theta \in[-60\degree,60\degree]" alt="\theta \in[-60\degree,60\degree]" class="ee_img tr_noresize" eeimg="1"> ，步长为 <img src="https://www.zhihu.com/equation?tex=30\degree" alt="30\degree" class="ee_img tr_noresize" eeimg="1"> 

 <img src="https://www.zhihu.com/equation?tex=s(t)表示M\times 1维的入射信号向量" alt="s(t)表示M\times 1维的入射信号向量" class="ee_img tr_noresize" eeimg="1"> ，即从CFAR的结果（某个Range Index和Velocity Index）提取的所有channel的值

 <img src="https://www.zhihu.com/equation?tex=x(t)=A^H*s(t)" alt="x(t)=A^H*s(t)" class="ee_img tr_noresize" eeimg="1"> 

 <img src="https://www.zhihu.com/equation?tex=A^H 表示A的共轭转置" alt="A^H 表示A的共轭转置" class="ee_img tr_noresize" eeimg="1"> 

 <img src="https://www.zhihu.com/equation?tex=x(t)" alt="x(t)" class="ee_img tr_noresize" eeimg="1"> 是 <img src="https://www.zhihu.com/equation?tex=D\times 1" alt="D\times 1" class="ee_img tr_noresize" eeimg="1"> 维的结果向量，相当于得到DBF(Digital Beamforming)谱，找角度最大值



## 俯仰角的解算（此处是针对的双物体）：

俯仰角的解算分为

+ 单目标
+ 双目标
+ 多目标

不同的场景采用不同的算法，对于单目标和双目标采用下面的算法即可

![image-20240310210520107](https://raw.githubusercontent.com/SendingA/Markdown4Zhihu/master/Data/毫米微波雷达4D点云成像的处理流程/image-20240310210520107.png)

![image-20240310210536842](https://raw.githubusercontent.com/SendingA/Markdown4Zhihu/master/Data/毫米微波雷达4D点云成像的处理流程/image-20240310210536842.png)

对于多目标， 使用DML(Deterministic Maximum Liklihood) or DBF-multi-obj

参考：[基于确定性最大似然算法 DML 的 DoA 估计，用牛顿法实现（附 MATLAB 源码）_dml sml-CSDN博客](https://blog.csdn.net/fengyanlover/article/details/130080596)
