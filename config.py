DATA_PATH = "../data/MLP_500_ver2.csv"
OUTPUT_DIR = "Result"

LR = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
WEIGHT_DECAY = [0.01, 0.005, 0.002, 0.001]
EARLY_STOP = [50]   # FIX
NUM_EPOCHS = [1000] # FIX
DROPOUT1 = [0.1, 0.2, 0.5]
DROPOUT2 = [0.1, 0.2, 0.5]
BATCH_SIZE = [32]   # FIX
HIDE1_NUM = [512]   # FIX
HIDE2_NUM = [256]   # FIX

PARAMS = [LR, WEIGHT_DECAY, EARLY_STOP, NUM_EPOCHS, DROPOUT1, DROPOUT2, BATCH_SIZE, HIDE1_NUM, HIDE2_NUM]