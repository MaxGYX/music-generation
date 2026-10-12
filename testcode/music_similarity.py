import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

def load_music_data(file_path):
    return pd.read_csv(file_path, low_memory=False)

def preprocess_features(data, feature_columns):
    scaler = StandardScaler()
    feature_data = data[feature_columns].values
    scaled_features = scaler.fit_transform(feature_data)
    return scaled_features
    # 将处理后的数据重新转换回 DataFrame 以便使用 dropna 方法
    # scaled_df = pd.DataFrame(scaled_features, columns=feature_columns)
    # scaled_df = scaled_df.dropna()
    # return scaled_df.values

def calculate_similarity_matrix(features):
    distance_matrix = euclidean_distances(features)
    similarity_matrix = 1 / (1 + distance_matrix)
    return similarity_matrix

def find_similar_music(music_id, similarity_matrix, data, top_n=10):
    similarity_scores = list(enumerate(similarity_matrix[music_id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_indices = [i[0] for i in similarity_scores[1:top_n + 1]]
    similar_scores = [i[1] for i in similarity_scores[1:top_n + 1]]
    similar_music = data.iloc[similar_indices].copy()
    similar_music['similarity_score'] = similar_scores
    return similar_music

# def visualize_music_clusters(features, data, n_clusters=8, sample_size=1000):
#     if len(data) > sample_size:
#         indices = np.random.choice(len(data), sample_size, replace=False)
#         sample_features = features[indices]
#         sample_data = data.iloc[indices].copy()
#     else:
#         sample_features = features
#         sample_data = data.copy()
#
#     # 根据样本数量调整 perplexity
#     perplexity_value = min(30, len(sample_data) - 1)
#     tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
#     reduced_features = tsne.fit_transform(sample_features)
#
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#     clusters = kmeans.fit_predict(sample_features)
#
#     plt.figure(figsize=(12, 10))
#     scatter = plt.scatter(
#         reduced_features[:, 0],
#         reduced_features[:, 1],
#         c=clusters,
#         cmap='viridis',
#         alpha=0.7,
#         s=50
#     )
#
#     plt.colorbar(scatter, label='Cluster')
#     plt.title('音乐聚类可视化')
#     plt.xlabel('t - SNE 1')
#     plt.ylabel('t - SNE 2')
#
#     centroids = kmeans.cluster_centers_
#     # 针对聚类中心的数量再次调整 perplexity
#     centroid_perplexity = min(30, len(centroids) - 1) if len(centroids) > 1 else 0
#     tsne = TSNE(n_components=2, random_state=42, perplexity=centroid_perplexity)
#     tsne_centroids = tsne.fit_transform(centroids)
#
#     for i, (cx, cy) in enumerate(tsne_centroids):
#         plt.scatter(cx, cy, marker='X', s=200, c='red')
#         plt.annotate(f'Cluster {i}', (cx, cy), fontsize=12, fontweight='bold')
#
#     plt.tight_layout()
#     plt.savefig('music_clusters.png')
#     print("聚类可视化已保存为 music_clusters.png")

def visualize_music_clusters(features, data, n_clusters=8, sample_size=1000):
    if len(data) > sample_size:
        indices = np.random.choice(len(data), sample_size, replace=False)
        sample_features = features[indices]
        sample_data = data.iloc[indices].copy()
    else:
        sample_features = features
        sample_data = data.copy()

    tsne = TSNE(n_components=3, random_state=42)
    reduced_features = tsne.fit_transform(sample_features)

    # 使用层次聚类
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = clustering.fit_predict(sample_features)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        reduced_features[:, 2],
        c=clusters,
        cmap='viridis',
        alpha=0.7,
        s=50
    )

    ax.set_xlabel('t - SNE 1')
    ax.set_ylabel('t - SNE 2')
    ax.set_zlabel('t - SNE 3')
    ax.set_title('音乐聚类可视化')

    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.savefig('music_clusters.png')
    plt.show()

if __name__ == "__main__":
    file_path = 'music_data.csv'
    feature_columns = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'mode',
                       'key', 'acousticness', 'instrumentalness', 'liveness', 'speechiness',
                       'explicit', 'duration_ms', 'popularity', 'year']

    music_data = load_music_data(file_path)
    processed_features = preprocess_features(music_data, feature_columns)
    similarity_matrix = calculate_similarity_matrix(processed_features)

    target_music_id = 0
    similar_music = find_similar_music(target_music_id, similarity_matrix, music_data, top_n=5)
    print(f"\n与音乐 ID {target_music_id} 相似的音乐:")
    print(similar_music[['similarity_score'] + feature_columns])

    visualize_music_clusters(processed_features, music_data)