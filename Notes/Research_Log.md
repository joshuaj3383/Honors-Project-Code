# Research Log

## Goal:


## Benchmark Model Structure:

Before testing over multiple hyperparameters, a benchmark model structure was required as a base.
Choosing the correct base architecture is vital as it needs to relatively small and trainable while also being accurate.
The model needs to be small for training efficiency and to avoid skewing results. 
It needs to be accurate as scaling laws require models to be trained near optimal loss. 

Based on paper (TODO CITATIONS) the recommended model architecture for speed and efficiency is a 
CNN which follows this architecture: 


From test results this model would score around 90% accuracy by around 50 epochs which is a good baseline as the model
sits at a relatively small 290,000 parameters (compared to most other benchmarked models).



## Data is needed to test the following:  
N: Number of Parameters in Model  
D: Number of Datatokens in dataset
C: Number of Compute operations (FLOPs)

### Scaling laws for N, D, C (Part A):
L(N,D,C) = Loss  
L(N) ~ N^-a  
L(D) ~ D^-b   
L(C) ~ C^-c

### Optimal N,D for fixed Compute C (Part B):

$L(N,D) = AN^{-a} + BD^{-b}$  
$C = k * N * D * e$, e = number epochs, k = some constant  
$L(N,D) = L(N) = AN^{-a} + B(C/kNe)^{-b}$   
$L(N) = AN^{-a} + B(ke/C)^{b} \cdot N^b$  

Optimize by taking the derivative and setting equal to 0

$L'(N) = -AaN^{-a-1} + Bb(ke/C)^{b} \cdot N^{b-1} = 0$  
$AaN^{-a-1} = Bb(ke/C)^{b} \cdot N^{b-1}$  
$Aa/Bb \cdot (C/ke)^b = N^{a+b}$  
$N^{a+b} \sim C^b$  
$N \sim C^{b/(a+b)}$

Using a similar approach with D we get:  

$D \sim C^{a/(a+b)}$

Following this logic:  

$N^{(a+b)/b} \sim C$  
$D \sim (N^{(a+b)/b})^{a/(a+b)}$  
$D \sim N^{a/b}$


This implies that given fixed compute, the optimal loss is provided when $D \sim N^{a/b}$  

This result will be tested in Part B using data collected in Part A

## Part A: Scaling Experiments

### N: Number of parameters:  
N can be scaled horizontally through the size of the channels. 
This is represented by width--the size of the output channel of the first block in the CNN.  
N can also be scaled depth-wise by changing the number of blocks per stage (before pooling).
This is not preferred over width-wise scaling as it changes the architecture of the model which has extra effects 
besides parameter amount which include:
- Changes in gradient descent as more non-linearity is introduced
- May require altered learning rates compared to a network with fewer layers
- Convolution reception becomes more abstracted the more layers which are added

For this reason, depth will be kept at the benchmark value of 2 for most of the tests. 
There will be a section with fixed width and varied depth for continuity purposes though.

**For N-Width:**  

    width = [8,16,32,64,128]  
    depth = [2]  
    dataset_size = [50000]  

**For N-Depth:**  

    width = [32]  
    depth = [1,2,3,4,5]  
    dataset_size = [50000]  

### D: Dataset Size:  

Width and depth are fixed to the benchmark values and dataset varies

    width = [32]  
    depth = [2]  
    dataset_size = [3125,6250,12500,25000,50000] 



### C: FLOPS

To scale for FLOPS, we need an array of N x D to fit the function L(N,D)

The first test was:

    width = [8,16,32,64]  
    depth = [2]  
    dataset_size = [6250,12500,25000,50000] 

After completing part B, I realized that this method varies dataset size while the fixed_compute testing varies epochs
so for continuity, I added a second test:

    width = [8,16,32,64]  
    depth = [2]  
    dataset_size = [50000] 
    epochs = [25,50,75,100]


## Part A: Results

After collecting training data, we needed to fit each dataset to the power-law to evaluate fit.

### 2D Regression

$L(X) = AX^{-a}+l_{base}$

|Title            |A                 |a                 |L_base            |r2                |
|-----------------|------------------|------------------|------------------|------------------|
|N-Width vs Loss  |51.120931473104605|0.4996066402541184|0.2366872426984081|0.9951657594926968|
|N-Depth vs Loss  |106505045.69140711|1.7450951628949545|0.2939159517935403|0.9964672906237726|
|D vs Loss        |8.53961896145369  |0.2235417515315491|-0.4265654848234568|0.9991062212414108|
|C (N-scaling) vs Loss|2844609.982905708 |0.5171566505174248|0.2383314538899608|0.9950286324212512|
|C (D-scaling) vs Loss|3763.0503987443544|0.26084446535682737|-0.29857166976457794|0.9989208180180228|


### 3D Regression
$L(N,D)=AN^{-a}+BD^{-b}+l_{base}$

| Title   |A                 |a                 |B                 |b                 |L_min                 |r2                |
|---------|------------------|------------------|------------------|------------------|----------------------|------------------|
| epochs  |68.09430544489504 |0.5317455135581164|4600913.072788369 |1.2742171278184249|0.2155034802575657    |0.9914078924892856|
| dataset |177.19604461031818|0.6570920219472594|26.213326449302752|0.4115841916762119|1.4043607535011016e-16|0.9861470131489078|


All of these show great fit (most r^2>.99) which suggests that the scaling law is applicable to medium-sized CCN's.


## Part B: Optimizing Over Fixed Compute

The goal of part B is compare the optimal N, D to the expected optimal values which are calculated from part A.

### Experimental method for Optimal Model

The first step to test under fixed compute and find the model which preforms the best.
To this, a baseline compute was taken from a model with Width=32,Depth=2,D=50000, and Epochs=50. 
From this, models with widths varying from 16 to 64 where trained with epochs calculated so that 
total compute would stay constant.
The step size started with 8 and was narrowed to 4 once the ideal region was found.
From this, the optimal model under the fixed compute was found to have a width of 36


### Mathematical method for Optimal Model

Having the experimentally tested optimal model, the next step was to see if the calculated values matched.
Using the values from the 3d regression and the formulas the optimal N, D for given C can be calculated as follows: 

$N = (\frac{aA}{bB})^{\frac{1}{a+b}} \cdot C^{\frac{b}{a+b}}$  
$D = (\frac{aA}{bB})^{\frac{-1}{a+b}} \cdot C^{\frac{a}{a+b}}$

## Part B: Results

The analysis code gave: 

    Tested optimal N,D tradeoff from: width=36.0, epochs=40.0
    N: 365518.0
    D: 2000000.0

    Optimal N,D tradeoff from test: Width vs Dataset size
    N: 342034.98294815706
    D: 2113775.011457192

    Optimal N,D tradeoff from test: Width vs Epochs
    N: 303987.5821754186
    D: 2378337.2821551524




