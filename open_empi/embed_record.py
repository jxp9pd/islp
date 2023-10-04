from sentence_transformers import SentenceTransformer
import ray

ray.init()

HF_MODEL = SentenceTransformer('all-mpnet-base-v2')
HF_MODEL_REF = ray.put(HF_MODEL)

@ray.remote
def remote_embed_record(record):
    """
    Returns a vector embedding for a single string

    record:parameter concatenated string
    embedding:return list of embedding for given concat string
    """
    model = ray.get(HF_MODEL_REF)
    return model.encode(record)

def shutdown_ray():
    ray.shutdown()
