import os
import pandas as pd
import argparse

from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument("--dir", required=True)
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--score", default="min_dist")

def main():
    args = parser.parse_args()
    
    df = pd.read_csv(os.path.join(args.dir, "data.csv"))

    X = df[[args.score]]

    kmeans = KMeans(n_clusters=args.n, random_state=0).fit(X)
    df['label'] = kmeans.labels_
    
    avg_configs = []
    for i in range(args.n):
        data = df[df.label == i]
        avg_configs.append(data.mean(axis=0))
    
    avg_df = pd.DataFrame(avg_configs)
    avg_df.to_csv(os.path.join(args.dir, "cluster.csv"))

if __name__ == "__main__":
    main()