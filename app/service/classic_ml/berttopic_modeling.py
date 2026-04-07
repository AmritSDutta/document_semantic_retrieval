import logging
from typing import List, Dict
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
import json

from app.config.Settings import Settings
from app.service.classic_ml.ml_helper import clean

settings = Settings()


def _get_data() -> List[str]:
    df = pd.read_json(settings.training_file_path, lines=True)
    df['clean'] = df['overall'].apply(clean)
    docs = df['clean'].astype(str).tolist()
    return docs


def _create_and_get_topic_model() -> BERTopic:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = UMAP(
        n_neighbors=10,
        n_components=5,
        min_dist=0.1,
        metric='cosine'
    )
    topic_model = BERTopic(
        umap_model=umap_model,
        embedding_model=embedding_model,
        verbose=True
    )
    return topic_model


def train_topic_model() -> Dict[str, str]:
    docs = _get_data()
    topic_model = _create_and_get_topic_model()

    topics, _ = topic_model.fit_transform(docs)

    # Build topic names
    topic_names = {
        t: get_topic_name(t, topic_model)
        for t in topic_model.get_topics().keys()
    }

    # Save model (safe format)
    topic_model.save("bertopic_model.pkl")

    # Save labels separately
    with open("topic_names.json", "w") as f:
        json.dump(topic_names, f)
    return topic_names


def infer_topic_model(new_docs):
    from bertopic import BERTopic
    import json

    topic_model = BERTopic.load("bertopic_model.pkl")

    with open("topic_names.json") as f:
        topic_names = json.load(f)

    new_topics, new_probs = topic_model.transform(new_docs)

    results = []
    for doc, t, p in zip(new_docs, new_topics, new_probs):
        label = topic_names.get(str(t), "Unknown")
        confidence = float(p.max()) if p is not None else 0.0

        results.append({
            "text": doc,
            "topic_id": int(t),
            "topic_name": label,
            "confidence": confidence
        })
    logging.info(f"classical ml topic results: {results}")
    return results


def get_topic_name(topic_id, topic_model):
    words = topic_model.get_topic(topic_id)
    return " ".join([w for w, _ in words[:3]])
