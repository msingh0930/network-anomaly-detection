from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import joblib

def train_random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, '../models/random_forest.pkl')
    return clf

def train_kmeans(X, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    joblib.dump(kmeans, '../models/kmeans.pkl')
    return kmeans