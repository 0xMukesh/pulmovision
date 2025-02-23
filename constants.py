from medmnist import INFO

# dataset related constants
DATASET_NAME = "chestmnist"
SHOULD_DOWNLOAD = True
DATASET_INFO = INFO[DATASET_NAME]
N_CHANNELS = DATASET_INFO["n_channels"]
N_CLASSES = len(DATASET_INFO["label"])
DATASET_CLASSES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
                   "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
                   "Edema", "Emphysema", "Fibrosis", "Pleural", "Hernia"]


# hyper-parameters
N_EPOCHS = 3
BATCH_SIZE = 128
LEARNING_RATE = 0.001
