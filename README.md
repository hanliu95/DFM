# Discrete Factorization Machines 

This is our experiments codes for the paper:

Han Liu, Xiangnan He, Fuli Feng, Liqiang Nie, Rui Liu, Hanwang Zhang (2018). Discrete Factorization Machines for Fast Feature-based Recommendation. In Proceedings of IJCAI'18.

*For any issue, please contact Han Liu: hanliu.sdu@gmail.com*


## Environment settings
MATLAB R2016a.

## Example to run the codes.
Please run the 'test.m' file by inputting the command below in MATLAB command window, then the training and the testing of DFM will automatically proceed.

Run DFM:
```
test
```
During the training process, the value of loss function and objective function will be printed in MATLAB command window after each optimization iteration.

Output (training process):
```
DFM at bit 64 Iteration:20
loss value = 1109405916.9047 obj value = 1153073221.0511
```
After the testing process, the NDCG(NDCG@1 to NDCG@10) of DFM on the testing set will be printed in MATLAB command window.

Output (testing process):
```
The DFM ndcg@1 is 0.81726
The DFM ndcg@2 is 0.812
The DFM ndcg@3 is 0.81073
The DFM ndcg@4 is 0.81256
The DFM ndcg@5 is 0.81639
The DFM ndcg@6 is 0.82228
The DFM ndcg@7 is 0.83002
The DFM ndcg@8 is 0.83914
The DFM ndcg@9 is 0.84877
The DFM ndcg@10 is 0.85742
```

### Dataset
We provide two processed datasets: Yelp and Amazon.

train_yelp, train_amazon:
- Train file.
- Each Line is a training instance: userID\itemID\rating

test_yelp, test_amazon:
- Test file.
- Each Line is a testing instance: userID\itemID\rating

feature_yelp, feature_amazon:
- Feature file.
- Each Line is a item's content-based information vector. 
