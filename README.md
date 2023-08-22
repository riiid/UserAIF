# Addressing Selection Bias in Computerized Adaptive Testing: A User-Wise Aggregate Influence Function Approach
Official Pytorch implementation of the paper "Addressing Selection Bias in Computerized Adaptive Testing: A User-Wise Aggregate Influence Function Approach"
## Requirements
``` 
pip install -r requirements.txt
``` 

## Training 
### 1) unbiasead + biased
``` 
python main_train.py
``` 
### 2) unbiased 
``` 
python main_train.py data.path_dict.biased_users=""
``` 

### 3) unbiasead + selected biased (UserAIF)
``` 
python main_train.py user_aif=True
``` 


