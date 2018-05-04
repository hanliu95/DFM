# Discrete Factorization Machines for Fast Feature-based Recommendation

This is our experiments codes for the paper:

Han Liu, Xiangnan He, Fuli Feng, Liqiang Nie, Rui Liu, Hanwang Zhang (2018). Discrete Factorization Machines for Fast Feature-based Recommendation. In Proceedings of IJCAI'18.


## Environment settings
We implement our proposed DFM method using MATLAB R2016a.

## Example to run the codes.
Please run the 'test.m' file.

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
