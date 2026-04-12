import logging
from typing import List, Dict, LiteralString
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from umap import UMAP
import json

from app.config.Settings import Settings, get_settings
from app.service.classic_ml.ml_helper import clean

logger = logging.getLogger(__name__)


def _get_data() -> List[str]:
    df = pd.read_json(get_settings().training_file_path, lines=True)
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
        hdbscan_model=KMeans(n_clusters=15),
        verbose=True
    )
    return topic_model


def _get_topic_name(topic_id, topic_model):
    words = topic_model.get_topic(topic_id)
    return " ".join([w for w, _ in words[:10]])


def train_topic_model() -> Dict[int, str]:
    logging.info("training topic model")
    docs = _get_data()
    logging.info("data gathered")

    topic_model = _create_and_get_topic_model()
    logging.info("initialize model")

    topics, _ = topic_model.fit_transform(docs)
    logging.info("training model finished")
    # Build topic names
    topic_names = {
        t: _get_topic_name(t, topic_model)
        for t in topic_model.get_topics().keys()
    }

    # Save model (safe format)
    topic_model.save("bertopic_model.pkl")
    logging.info("saved trained model parameters")

    # Save labels separately
    with open("topic_names.json", "w") as f:
        json.dump(topic_names, f)

    logging.info(f"Topics discovered: {topic_names}")

    return topic_names


def infer_topic_model(new_docs):
    from bertopic import BERTopic
    import json
    logging.info("training model getting loaded")
    topic_model = BERTopic.load("bertopic_model.pkl")
    logging.info("training model loaded")

    with open("topic_names.json") as f:
        topic_names = json.load(f)
    logging.info(f"Topics discovered: {topic_names}")

    new_topics, _ = topic_model.transform(new_docs)
    assert new_topics is not None

    results = []
    for doc, t in zip(new_docs, new_topics):
        label = topic_names.get(str(t), "Unknown")
        confidence = 1.0

        results.append({
            "text": doc,
            "topic_id": int(t),
            "topic_name": label,
            "confidence": confidence
        })
    logging.info(f"classical ml topic results: {results}")
    return results
