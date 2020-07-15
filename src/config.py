# where the data is stored
RAW_DIR="data/raw"
PROC_DIR="data/processed"
TRAIN_DATA="data/train_temp.csv"
VALID_DATA="data/valid_temp.csv"
TEST_DATA="data/test_proc.csv"

# training params
N_FOLDS=2
TEST_SIZE=0.1
TRAIN_BATCH_SIZE=10
VALID_BATCH_SIZE=25
RAND_STATE=1212