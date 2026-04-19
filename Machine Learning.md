## Definition
Fields of study that gives computers the ability to learn without being explicitly programmed

## Terminology
### Training Set
Data used to train the model
### Notation(数学符号)
x = “input” variable 
	feature
y = output variable 
	target variable

m = number of training examples

(x , y) = single training example

$$ (x^{(i)}, y^{(i)}) = i^{\text{th}} \text{ training example } (1^{\text{st}}, 2^{\text{nd}}, 3^{\text{rd}} \dots) $$
f = function(Model)

$\hat{y}$ = predict value

### Cost Function(代价函数)

Squared error cost function:is the most commonly used one for linear regression

$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2$$
- $J(w,b)$:代价函数。它是一个标量，用来衡量当前模型预测误差有多大。我们的目标是找到一组w和b，使J的值最小
- m：训练样本的总数量
- **$\sum_{i=1}^{m}$**:求和符号，表示将所有m个样本的误差累加起来
- **$f_{w,b}(x^{(i)})$**:模型对第i个样本的预测值（通常是 $w \cdot x^{(i)} + b$）。
- **$y^{(i)}$**: 第 $i$ 个样本的**真实标签**（实际结果）。
- **$(f_{w,b}(x^{(i)}) - y^{(i)})^2$**: 预测值与真实值之差的平方，即“平方误差”。使用平方是为了确保误差总是正数，并放大较大的误差。
- **$\frac{1}{2m}$**:

	- 除以 $m$ 是为了取**平均误差**，这样代价函数的值就不会随着样本数量的增加而剧烈波动。
    
	- 乘以 $\frac{1}{2}$ 是为了在后续求导（计算梯度）时，抵消掉平方项下落产生的系数 $2$，让求导后的式子更简洁。

 
## Supervised learning

Learns from being given "**right answers**"
learning input-output or X to Y mappings

### Types

#### 1. Regression Algorithm

Predict a number 
infinitely many possible outputs
example: predict house prices

##### 1.1 Linear Regression Model

Linear regression with one variable

Also named univariate linear regression

###### Multiple features
- $x_j$ = $j^{th}$ feature 
- $n$ = number of features 
- $\vec{x}^{(i)}$ = features of $i^{th}$ training example
$f_{w,b}(x) = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$
$f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$

 

#### 2. Classification Algorithm

Classification predict categories
small number of possible outputs
example:breast cancer detection

## Unsupervised learning

Find something interesting in unlabeled data
Data only comes with inputs x, but not output labels y
Algorithm has to find structure in the data

### Types

#### 1. Clustering

Group similar data points together
Takes date without labels and tries to automatically group them into clusters
example: Google News、Grouping Customers

#### 2. Anomaly Detection(异常检测)

Find unusual data points

#### 3. Dimensionality reduction（降维）

Compress data using fewer numbers


## Reinforcement learning


## Gradient descent （梯度下降）

### Definition
have some function $J(w,b)$

want $\min_{w,b} J(w,b)$

### Outline
- Start with some w,b
- keep changing w,b to reduce $J(w,b)$
- Until we settle at or near a minimum

### Gradient descent algorithm

$$w = w - \alpha \frac{\partial}{\partial w} J(w,b)$$
- w: weight 权重函数
- =：赋值
- α：learning rate，学习率，控制每一步更新的步长
- $\frac{\partial}{\partial w}$ :偏导数符号，表示对w求导
- $J(w,b)$ cost function,代价函数，衡量模型预测值与真实值之间的差距
- $\frac{\partial}{\partial w} J(w,b)$ 梯度，他指明了函数值上升最快的方向，我们在前面加了负号，所以是朝着函数下降最快的方向迈进

$$b = b - \alpha \frac{\partial}{\partial b} J(w,b)$$
- b:偏置项（Bias）
- α：学习率（learning rate）
- $\frac{\partial}{\partial b} J(w,b)$:代价函数J对偏置b的偏导数，即在b方向上的梯度


### Learning Rate

Control how big of a step you take when updating the model's parameters w and b

 If the α is too small:
	 Gradient descent may be slow
 If the α is too large:
	 Gradient  descent may overshoot, never reach minimum
	 failed to converge, diverge

### 1、Batch gradient descent

Batch: each step of gradient descent uses all the training examples
- 每一轮迭代都使用全部训练样本来计算梯度并更新参数
- **计算公式**：$\theta = \theta - \alpha \frac{1}{m} \sum_{i=1}^{m} \nabla_{\theta} J_i(\theta)$
- **内存占用**：**极高**。必须一次性将整个数据集读入内存，否则无法完成求和运算。
- **优点**：梯度下降路径非常平滑，能准确地朝着全局最小值前进。
- **缺点**：数据量大时速度极慢，且容易因内存不足（OOM）导致程序崩溃。
### 2、Stochastic GD，SGD
- **概念**：每一轮迭代只随机选择**一个**样本来计算梯度并立即更新参数。
- **计算公式**：$\theta = \theta - \alpha \nabla_{\theta} J_i(\theta)$
- **内存占用**：**极低**。内存中永远只保留一个样本的数据，对硬件要求最低。
- **优点**：更新速度极快；随机性有助于跳出局部最小值（Local Minimum）。
- **缺点**：下降路径非常“震荡（Noisy）”，即使到了最小值附近也会来回跳动，很难真正稳定在最优点。
### 3、Mini-batch GD
- **概念**：折中方案。将数据集分成很多小份（称为 Batch），每次迭代只使用其中一小部分（通常是 32, 64, 128 或 256 条数据）。
- **计算公式**：$\theta = \theta - \alpha \frac{1}{\text{batch\_size}} \sum_{i \in \text{Batch}} \nabla_{\theta} J_i(\theta)$
- **内存占用**：**可控且稳定**。内存占用仅取决于你设置的 `batch_size`，与总数据量无关。
- **优点**：
    1. **工业标准**：兼具 BGD 的稳定性和 SGD 的效率。
    2. **硬件加速**：可以利用 GPU 的并行计算能力同时处理这一个小批次的数据。
- **缺点**：需要额外调节一个参数——`batch_size`（批大小）。

## Vectorization
### Parameters and features
$f_{\vec{w},b}(\vec{x}) = w_1 x_1 + w_2 x_2 + w_3 x_3 + b$
- $\vec{w} = \begin{bmatrix} w_1 & w_2 & w_3 \end{bmatrix}$ 
- $b$ is a number 
- $\vec{x} = \begin{bmatrix} x_1 & x_2 & x_3 \end{bmatrix}$ 
- linear algebra: count from 1
 





 
 
