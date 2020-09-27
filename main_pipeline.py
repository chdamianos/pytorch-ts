from pytorch_ts import features
import pandas as pd
train_pdf = pd.read_csv('./data/train.csv')

train_features_instance = features.training_features.TrainingFeatures()
train_features_pdf = train_features_instance.create_training_features(
    train_pdf)
train_features_pdf.to_pickle(
    "./data/features_reduced_memory_pdf_step5_to_15_main_2")
