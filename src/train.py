from preprocess import load_data, preprocess_data
from model import train_random_forest

df = load_data("../data/large_data_log.csv")
df = preprocess_data(df)

X = df.drop("label", axis=1)
y = df["label"]

train_random_forest(X, y)
print("Model trained and saved to models/random_forest.pkl")