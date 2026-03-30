from playwright.sync_api import sync_playwright, Playwright, Page, Request
from sentence_transformers import SentenceTransformer
import chromadb
from datetime import datetime
import requests
from uuid import uuid4
from bs4 import BeautifulSoup

from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from collections import Counter, defaultdict
import numpy as np

# Rough idea of pipeline...
# 1. Have the model describe each page based on the html content
# 2. Record the following to an embedding database:
#       - Summary
#       - Stripped HTML Content
#       - Meta Data: Time, URL, Time Spent on Page
# 3. Update embedding clusters
# 4. Summarize the clusters
# 5. Write to .md file

LOCAL_LLM_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3:8b"
TIMESTAMP_FMT = "%m/%d/%Y %H:%M:%S"
GENERAL_PROMPT = "Your job is to create diary entries describing a user's workflow on their desktop."

EVENTS_PROMPT = f"""
Your job is to create a single diary entry for a diary conversion pipeline. As an auxilary model, you must only respond with the exact text specified; it will be ingested by a machine for postprocessing. No acknowledgement, no followup questions, no additional formatting. Just the key points. Period.\n
The verbose entry log you receive will already have been clustered for semantic similarity, so you must do your best to find generalizable diary title encompassing the event activities.\n
The diary entry must follow a strict format:\n\n

[Diary entry topic]\n
[Summarized activity]\n
[Key resources the human used]\n
[Optional IFF enough info given: inferred intent or outcome]\n\n

Here is an example of how your response should be structured:
**Ebay Homepage Analysis**\n
Summarized activity: Analyzed JSON response from eBay's homepage\n
Key resources used: eBay's homepage data, JSON object\n
Intent: Page rendering and tracking metadata extraction\n\n

"""

def remove_tags(html: str) -> str:
    """
    Returns a webpages content, stripped of html tags and javascript code snippets.
    
    Args:
        html (str): The raw html content of a webpage.
        
    Returns:
        str: The webpages content, stripped of html tags and javascript code snippets.
    """
    # parse html content
    soup = BeautifulSoup(html, "html.parser")

    for data in soup(['style', 'script']):
        # Remove tags
        data.decompose()

    # return data by retrieving the tag content
    return ' '.join(soup.stripped_strings)

def update_all(collection, clusters_collection, distance_threshold=1.0):
    """
    Updates all items in the collection by clustering them and updating the clusters collection.
    
    Args:
        collection: The collection of items to cluster.
        clusters_collection: The collection of clusters to update.
        distance_threshold (float): The distance threshold for clustering.
        
    Returns:
        NoneType
    """
    
    # retrieve all items to cluster them
    results = collection.get(include=['embeddings', 'metadatas'])
    if not results['ids']:
        return

    embeddings = np.array(results['embeddings'])
    ids = results['ids']
    metadatas = results['metadatas']

    # agglomerative clustering creates clusters dynamically based on distance limit
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='euclidean',
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings)

    new_metadatas = []
    cluster_data = {}

    # 
    for i, label in enumerate(labels):
        meta = metadatas[i] if metadatas[i] else {}
        meta['cluster_id'] = int(label)
        new_metadatas.append(meta)

        if label not in cluster_data:
            cluster_data[label] = []
        cluster_data[label].append(embeddings[i])

    # add cluster identifier to metadata of text embeddings
    collection.update(ids=ids, metadatas=new_metadatas)

    # full refresh of centroids
    existing_clusters = clusters_collection.get()
    if existing_clusters['ids']:
        clusters_collection.delete(ids=existing_clusters['ids'])

    cluster_ids = []
    centroids = []
    for c_id, embs in cluster_data.items():
        cluster_ids.append(str(c_id))
        centroids.append(np.mean(embs, axis=0).tolist())
    
    clusters_collection.add(ids=cluster_ids, embeddings=centroids)
    
    


def update(collection, clusters_collection, embed_vector, embed_mdata, embed_text, distance_threshold=1.0):
    new_id = str(uuid4()) # generate unique identifier
    existing_clusters = clusters_collection.get(include=['embeddings'])
    assigned_cluster_id = -1

    # search for matching centroid / prexisting cluster
    if existing_clusters['ids']:
        res = clusters_collection.query(
            query_embeddings=[embed_vector.tolist() if isinstance(embed_vector, np.ndarray) else embed_vector],
            n_results=1
        )
        if res['distances'] and res['distances'][0]:
            dist = res['distances'][0][0]
            if dist < distance_threshold:
                assigned_cluster_id = int(res['ids'][0][0])

    embed_mdata['cluster_id'] = assigned_cluster_id
    
    # upsert to text embed collection
    collection.upsert(
        ids=[new_id],
        embeddings=[embed_vector],
        documents=[embed_text],
        metadatas=[embed_mdata]
    )

    # if new embedding is an outlier, compare it with other outliers, try to cluster it
    if assigned_cluster_id == -1:
        # retreive all outlier embeds
        unclustered_res = collection.get(
            where={"cluster_id": -1},
            include=['embeddings', 'metadatas']
        )

        unclustered_ids = unclustered_res['ids']
        unclustered_embs = unclustered_res['embeddings']

        # recluster if 2 or more outliers to compare
        if len(unclustered_ids) > 1:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric='euclidean',
                linkage='average'
            )
            labels = clustering.fit_predict(unclustered_embs)

            max_cluster_id = max([int(cid) for cid in existing_clusters['ids']]) if existing_clusters['ids'] else -1

            new_metadatas = []
            new_centroids = {}

            for i, label in enumerate(labels):
                new_global_label = max_cluster_id + 1 + label
                meta = unclustered_res['metadatas'][i]
                meta['cluster_id'] = int(new_global_label)
                new_metadatas.append(meta)

                if new_global_label not in new_centroids:
                    new_centroids[new_global_label] = []
                new_centroids[new_global_label].append(unclustered_embs[i])

            # update outlier metadata in text embed collection
            collection.update(ids=unclustered_ids, metadatas=new_metadatas)

            # make new centroid and add it to the collection
            c_ids = []
            c_embs = []
            for c_id, embs in new_centroids.items():
                c_ids.append(str(c_id))
                c_embs.append(np.mean(embs, axis=0).tolist())

            clusters_collection.add(ids=c_ids, embeddings=c_embs)

def get_cluster_event_mdata(collection) -> dict:
    results = collection.get(include=['metadatas'])

    clusters = {}
    
    if not results['metadatas']:
        return {}
        
    for mdata in results['metadatas']:
        if not mdata: continue
        cluster_id = mdata.get('cluster_id', -1)
        summary = mdata.get('summary', '')
        if cluster_id not in clusters:
            clusters[cluster_id] = []
            
        clusters[cluster_id].append(mdata)
        
    return clusters
    
def get_verbose_event_logs(event_mdata):
    text = ""
    
    for mdata in event_mdata:
        text_fragment = f"""
[Start of Entry]\n
Date: {mdata.get('time_start', 'n/a')} - {mdata.get('time_end', 'n/a')}\n
Time Spent: {mdata.get('time_spent', "n/a")}
Page Title: {mdata.get('page_title', 'n/a')}\n
Page URL: {mdata.get('page_url', 'n/a')}\n
AI Summary of the Page: {mdata.get('summary', '')}\n
[End of Entry]\n\n
        """
        text += text_fragment
        
    return text
        
def get_journal_timeframe(event_mdata):
    min_timestamp = None
    max_timestamp = None
    total_mins = 0
    
    for mdata in event_mdata:
        timestamp = datetime.strptime(mdata.get('time_start', 'n/a'), TIMESTAMP_FMT)
        if not min_timestamp:
            min_timestamp = timestamp
        
        if timestamp < min_timestamp:
            min_timestamp = timestamp
            
        timestamp = datetime.strptime(mdata.get('time_end', 'n/a'), TIMESTAMP_FMT)
        if not max_timestamp:
            max_timestamp = timestamp
        
        if timestamp > max_timestamp:
            max_timestamp = timestamp
            
        total_mins += int(mdata.get('time_spent', 0))
            
    min_timestamp_str = datetime.strftime(min_timestamp, TIMESTAMP_FMT) if min_timestamp else "Unknown"
    max_timestamp_str = datetime.strftime(max_timestamp, TIMESTAMP_FMT) if max_timestamp else "Unknown"
    journal_timeframe = f"{min_timestamp_str} - {max_timestamp_str} (Total Mins: {total_mins})"
    return journal_timeframe
        

def generate_diary(collection):
    grouped_mdata = get_cluster_event_mdata(collection)
    
    texts = []
    for mdata_list in grouped_mdata.values():
        entry_events_text = get_verbose_event_logs(mdata_list)
        entry_timeframe = get_journal_timeframe(mdata_list)
        entry_content = ask_llm(f"{EVENTS_PROMPT}\n\nVerbose Event Logs:\n{entry_events_text}\n")
        
        entry_text = f"{entry_timeframe}\n{entry_content}"
        print(entry_text)
        texts.append(entry_text)
        
    with open("journal.txt", "w") as f:
        sep = f"\n\n{'-'*20}\n\n"
        f.write(sep.join(texts))
        print("Journal successfully saved")
    

def get_clustering_info(collection):
    print("\n--- Cluster Summary ---")
    results = collection.get(include=['metadatas'])

    if results['metadatas']:
        # Safely extract cluster_ids, defaulting to -1 if missing for some reason
        cluster_ids = [meta.get('cluster_id', -1) for meta in results['metadatas'] if meta]

        # Tally up the cluster sizes
        counts = Counter(cluster_ids)

        total_embeds = len(cluster_ids)
        total_clusters = len(counts)

        # Isolate clusters that only have 1 embedding (our functional "outliers")
        outliers = {cid: count for cid, count in counts.items() if count == 1}

        print(f"Total Embeddings: {total_embeds}")
        print(f"Total Clusters: {total_clusters}")
        print(f"Total Outliers (Clusters of size 1): {len(outliers)}\n")

        print("Embeds per Multi-Item Cluster:")
        # .most_common() sorts by size descending
        for cid, count in counts.most_common():
            if count > 1:
                print(f"  Cluster {cid}: {count} items")

        if outliers:
            print(f"\nOutlier Cluster IDs (1 item each):")
            # Print just the keys (the cluster IDs) since we know the count is 1
            print(f"  {list(outliers.keys())}")
    else:
        print("Collection is empty. No clusters to display.")
        


def ask_llm(prompt: str):
    response = requests.post(LOCAL_LLM_URL, json={
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    })
    return response.json().get("response", "").strip()

def get_page_summary(title, content):
    strip_content = remove_tags(content)
    prompt = f"Summarize the following page:\n\nTitle: {title}\n\nContent: {strip_content}"
    return ask_llm(prompt)



if __name__ == "__main__":
    chroma_client = chromadb.PersistentClient(path="./browser_memory_db")
    collection = chroma_client.get_or_create_collection(name="browser_events")
    generate_diary(collection)
