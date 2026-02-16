# Research Log

## Benchmark Model Structure:


## Data is needed to test the following:  
Scaling laws for N, D, C  
Optimal N,D for fixed Compute C

**These are parameters to get data:**

    width = [8,16,32,64,128]  
    depth = [1,2,3]  
    dataset_size = [6250,12500,25000,50000]  
    seed_choices = [0,1]

    epochs = 100  
    batch_size = 512


Training a model for each of the hyperparameters would require about 120 models with ~15m training time on average.
Since this is impractical and unnecessary, each law can be isolated and tested with smaller parameter range

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
    depth = [1,2,3]  
    dataset_size = [50000]  

### D: Dataset Size:  

Width and depth are fixed to the benchmark values and dataset varies

    width = [32]  
    depth = [2]  
    dataset_size = [6250,12500,25000,50000] 

