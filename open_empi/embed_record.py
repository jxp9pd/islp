from sentence_transformers import SentenceTransformer
import ray

HF_MODEL = SentenceTransformer('all-mpnet-base-v2')

@ray.remote
def remote_embed_record(model, record):
    """
    Returns a vector embedding for a single string

    record:parameter concatenated string
    embedding:return list of embedding for given concat string
    """

    embedding = model.encode(record)
    return embedding
