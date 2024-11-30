import numpy as np
import random
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances,cosine_similarity
from collections import defaultdict
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import spacy
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn import mixture
import math
import os

# Load NLP models

class LLMSelectorEnv:
    def __init__(self, llms,costs, eps=0.59, min_samples=1,n_clusters=1):
        self.total_clusters = n_clusters
        self.query_embeddings = None
        self.nlp = spacy.load("en_core_web_sm")
        self.llms = llms  # List of LLMs
        self.llm_names = list(llms.keys())
        self.n_llms = len(llms)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Embedding model
        self.q_table = None  # Q-table
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=42)  # Query clustering
        self.query_clusters = {}
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.1
        self.costs = costs
        self.predictor = KNeighborsClassifier(n_neighbors=4)  # Use 1-NN for simplicity


    def fit_clusters(self, queries):
        self.query_embeddings = self.model.encode(queries)
        self.query_clusters = self.clusterer.fit_predict(self.query_embeddings)
        self.update_clusters(queries)
        if os.path.exists('q_val.logs'):
            with open('q_val.logs', 'r') as f:
                f.readline()  # Skip the first line
                self.q_table = np.loadtxt(f)
        else:
            self.q_table = np.zeros((self.total_clusters, self.n_llms))

    def update_clusters(self,queries):
        all_seperated = False
        while not all_seperated:
            cluster_assignments = self.query_clusters
            clusters = defaultdict(list)
            for query, cluster_id in zip(queries, cluster_assignments):
                clusters[cluster_id].append(query)
            all_seperated = True
            fault_cluster = -1
            for i in range(self.total_clusters):
                if len(self.query_clusters[self.query_clusters == i]) > 1 and math.sqrt((np.sum(np.square(np.mean(self.query_embeddings[self.query_clusters == i]) - self.query_embeddings[self.query_clusters == i])))/self.total_clusters) > 1.3:
                    print("Faulty Cluster:",i,"Mean:",np.mean(self.query_embeddings[self.query_clusters == i]),"Varience:",np.sum(np.square(np.mean(self.query_embeddings[self.query_clusters == i]) - self.query_embeddings[self.query_clusters == i])))
                    all_seperated = False
                    fault_cluster = i
                    break
            if not all_seperated:
                fault_cluster_queries = clusters[fault_cluster]
                new_clusterer = KMeans(n_clusters=2, random_state=42)
                new_labels = new_clusterer.fit_predict(self.model.encode(fault_cluster_queries))
                f_index = 0
                is_seperated = False
                for i in range(len(self.query_clusters)):
                    if self.query_clusters[i] == fault_cluster:
                        if(new_labels[f_index] == 1):
                            self.query_clusters[i] = self.total_clusters
                            is_seperated = True
                        f_index += 1
                if is_seperated:
                    self.total_clusters += 1
        self.clusterer = KMeans(n_clusters=self.total_clusters, random_state=42)
        self.query_clusters = self.clusterer.fit_predict(self.query_embeddings)
        print(self.query_clusters)

        
    def get_state(self, query):
        embedding = self.model.encode([query])
        np.append(self.query_embeddings, embedding)
        predicted_cluster = self.clusterer.predict(embedding)
        np.append(self.query_clusters,predicted_cluster)
        return predicted_cluster

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:
            return random.sample(range(self.n_llms), 2)  # Explore
        else:
            # Exploit by choosing top 2 LLMs with highest Q-values
            return np.argsort(-self.q_table[state])[:2].tolist()
        
    def get_best_model(self, state):
        return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward):
        # Update Q-values using the Bellman equation with cost weightage
        best_next_action = np.max(self.q_table[state])
        for act in action:
            cost_weighted_reward = (0.8 * reward) - (0.2 * self.costs[act])
            self.q_table[state, act] += self.alpha * (cost_weighted_reward + self.gamma * best_next_action - self.q_table[state, act])
        self.print_q_table()

    def visualize_embeddings(self):
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(self.query_embeddings)
        unique_clusters = np.unique(self.query_clusters)
        colors = plt.cm.get_cmap('tab10', len(unique_clusters))

        for cluster_id in unique_clusters:
            cluster_points = reduced_embeddings[self.query_clusters == cluster_id]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id + 1}', color=colors(cluster_id))

        plt.legend()
        plt.title('Embedding Visualization with t-SNE')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.show()
        
    def get_cluster_topic(self,queries):
        keywords = []
        for query in queries:
            doc = self.nlp(query)
            keywords += [token.text.lower() for token in doc if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop]

        unique_keywords = list(set(keywords))
        if not unique_keywords:
            return "General"

        keyword_embeddings = self.model.encode(unique_keywords)
        similarity_matrix = cosine_similarity(keyword_embeddings)
        avg_similarity = similarity_matrix.mean(axis=0)
        top_indices = np.argsort(avg_similarity)[-2:]
        top_keywords = [unique_keywords[i] for i in top_indices]
        
        return " & ".join(top_keywords)

    # Print clusters with topics
    def print_clusters(self, queries):
        print("$-----Cluster-Started-----$\n")
        cluster_assignments = self.query_clusters
        clusters = defaultdict(list)

        for query, cluster_id in zip(queries, cluster_assignments):
            clusters[cluster_id].append(query)

        for cluster_id, cluster_queries in clusters.items():
            if cluster_id == -1:
                print(f"\nCluster {cluster_id + 1} (Noise):")
            else:
                topic = self.get_cluster_topic(cluster_queries)
                print(f"\nCluster {cluster_id + 1} ({topic}):")
            for query in cluster_queries:
                print(f"  - {query}")
        print("\n$-----Cluster-Ended-----$")
    def print_q_table(self):
        with open('q_val.logs', 'w') as f:
            f.write("Q-table:\n")
            np.savetxt(f, self.q_table)