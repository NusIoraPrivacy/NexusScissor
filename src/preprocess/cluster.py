from sklearn.cluster import AgglomerativeClustering

from sentence_transformers import SentenceTransformer

import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default='../../')
    parser.add_argument("--config_cache_path", type=str, default='config')
    parser.add_argument("--model_name", type=str, default='LLAMA27B')
    parser.add_argument("--data_name", type=str, default='advbench', choices=['advbench', 'FQ'])
    parser.add_argument("--level", type=str, default="first_level", choices=["first_level", "all_level"])
    parser.add_argument("--st_transformer", type=str, default="all-mpnet-base-v2", 
        choices=["Alibaba-NLP/gte-Qwen1.5-7B-instruct", "all-MiniLM-L6-v2", "all-mpnet-base-v2"])
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    embedder = SentenceTransformer(args.st_transformer)
    # Corpus with example sentences
    data_path = f"{args.root_path}/result/finetune/{args.model_name}/{args.level}/harmfulResp_harmful_sentences_{args.data_name}.json"
    with open(data_path) as f:
        harmful_sents = json.load(f)
    
    all_sents = []
    all_prompts = []
    for prompt, sents in harmful_sents.items():
        for sent in sents:
            all_sents.append(sent)
            all_prompts.append(prompt)
    
    corpus_embeddings = embedder.encode(all_sents)

    # Some models don't automatically normalize the embeddings, in which case you should normalize the embeddings:
    # corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    # Perform agglomerative clustering
    clustering_model = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=1.5 # 1.5
    )  # , affinity='cosine', linkage='average', distance_threshold=0.4)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        cluster_id = str(cluster_id)
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append([all_prompts[sentence_id], all_sents[sentence_id]])

    model_name = args.st_transformer.split("/")[-1]
    out_path = f"{args.root_path}/result/finetune/{args.model_name}/{args.level}/cluster_{model_name}_harmful_sentences_{args.data_name}.json"
    with open(out_path, "w") as f:
        json.dump(clustered_sentences, f, indent=4)

    print(len(clustered_sentences))
    print(len(clustered_sentences)/len(all_sents))