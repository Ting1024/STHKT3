D_MODEL=128

HEADNUM=8

BATCH_SIZE=50
MAX_EPOCH=50
MIN_SEQ=15
MAX_SEQ=200
LR=0.1

CHAIN_LEN=9

DEVICE="cuda:0"

HAS_PREG=False
HAS_HAWKES=True
ABLATION="withoutHawkes" #"withoutConv" #"withoutHawkes"

##############################DATASET PARAMS################################

DATASET="Aaai23"
if DATASET=="Assistment2009":
    DATASET_ROOTPATH="/data/Ting123/Assist2009/"
    NUM_Q=17751
    NUM_KC=123
    NUM_STU=4163
elif DATASET=="Assistment2017":
    DATASET_ROOTPATH="/data/Ting123/assist2017/"
    NUM_Q=4117 
    NUM_KC=102 
    NUM_STU=1709 
elif DATASET=="Stat2011":
    DATASET_ROOTPATH="/data/Ting123/stat2011/"
    NUM_Q=663
    NUM_KC=80
    NUM_STU=331
elif DATASET=="Aaai23":
    DATASET_ROOTPATH="/data/Ting123/Aaai23/"
    NUM_Q=7652
    NUM_KC=865
    NUM_STU=18066

TRAIN_FILE=DATASET_ROOTPATH+"normdata/traindata.csv"
TEST_FILE=DATASET_ROOTPATH+"normdata/testdata.csv"

DICT_Q2INDEX=DATASET_ROOTPATH+"normdata/dict/q2index.pkl"
DICT_INDEX2Q=DATASET_ROOTPATH+"normdata/dict/index2q.pkl"
DICT_KC2INDEX=DATASET_ROOTPATH+"normdata/dict/kc2index.pkl"
DICT_INDEX2KC=DATASET_ROOTPATH+"normdata/dict/index2kc.pkl"

MATRIX_QKC=DATASET_ROOTPATH+"normdata/matrix/M_qkc.pt"
MATRIX_KCS=DATASET_ROOTPATH+"normdata/matrix/M_kcs.pt"
MATRIX_QS=DATASET_ROOTPATH+"normdata/matrix/M_qs.pt"
MATRIX_QQ=DATASET_ROOTPATH+"normdata/matrix/M_qq.pt"
MATRIX_KK=DATASET_ROOTPATH+"normdata/matrix/M_kk.pt"
MATRIX_SS=DATASET_ROOTPATH+"normdata/matrix/M_ss.pt"


##########################################################################


MODEL_SAVE_FOLDER="/data/Ting123/assist2017/densetest_dataset/model/"
SAVEMODEL=MODEL_SAVE_FOLDER+"model.pt"
PIC_SAVE_FOLDER="/code/HawkesGKT/saving/picture/"

# import os
# if __name__=="__main__":
#     if not os.path.exists(MODEL_SAVE_FOLDER):
#         os.makedirs(MODEL_SAVE_FOLDER)
