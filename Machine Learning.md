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

