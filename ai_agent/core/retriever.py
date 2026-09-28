# import logging
# import time
# from typing import List
# from pydantic import BaseModel
# from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
# from langchain_huggingface import HuggingFaceEmbeddings
# from sentence_transformers import CrossEncoder

# from ai_agent.core.config import settings

# log = logging.getLogger(__name__)


# class RetrievedDocumentChunk(BaseModel):
#     chunk_text: str
#     chunk_start_index: int
#     source_document: str


# class Retriever:
#     def __init__(self):
#         connections.connect(
#             alias="default",
#             host=settings.MILVUS_HOST,
#             port=settings.MILVUS_PORT
#         )

#         self.embeddings = HuggingFaceEmbeddings(model_name=settings.MODEL_NAME)
#         self.collection = Collection(settings.MILVUS_COLLECTION)
#         self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

#     def start(self):
#         self.collection.load()
#         log.info("Milvus collection loaded into memory")

#     def shutdown(self):
#         self.collection.release()
#         log.info("Milvus collection released from memory")

#     def delete_collection(self):
#         """Delete the Milvus collection if it exists."""
#         if utility.has_collection(settings.MILVUS_COLLECTION):
#             utility.drop_collection(settings.MILVUS_COLLECTION)
#             log.info(f"Successfully deleted collection: {settings.MILVUS_COLLECTION}")
#         else:
#             log.info(f"Collection {settings.MILVUS_COLLECTION} does not exist")


# class VectorDBQuery:
#     def __init__(self):
#         connections.connect(
#             alias="default",
#             host=settings.MILVUS_HOST,
#             port=settings.MILVUS_PORT
#         )

#         self.embeddings = HuggingFaceEmbeddings(model_name=settings.MODEL_NAME)
#         self.collection = Collection(settings.MILVUS_COLLECTION)
#         self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

#     def query(
#         self,
#         query: str,
#         type: str = "content",   # "content" or "metadata"
#         initial_candidates: int = 20,
#         final_top_k: int = 10
#     ):
#         """
#         If type=content → return dict {"free": str, "paid": str} with text.
#         If type=metadata → return dict {"free": [..], "paid": [..]} with structured metadata (filename, filepath, url).
#         """
#         self.collection.load()
#         query_embedding = self.embeddings.embed_query(query)

#         search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

#         results = self.collection.search(
#             data=[query_embedding],
#             anns_field="embedding",
#             param=search_params,
#             limit=initial_candidates,
#             output_fields=["text", "filename", "filepath", "start_index", "access_type", "type", "url"]
#         )

#         # --- If metadata query ---
#         if type == "metadata":
#             candidates = []
#             for hits in results:
#                 for hit in hits:
#                     if hit.entity.get("type") != "metadata":
#                         continue
#                     candidates.append({
#                         "filename": hit.entity.get("filename"),
#                         "filepath": hit.entity.get("filepath"),
#                         "access_type": (hit.entity.get("access_type") or "free").lower(),
#                         "url": hit.entity.get("url")
#                     })

#             grouped = {"free": [], "paid": []}
#             for c in candidates[:final_top_k]:
#                 grouped[c["access_type"]].append(c)

#             return grouped

#         # --- If content query ---
#         candidates = []
#         for hits in results:
#             for hit in hits:
#                 text = hit.entity.get("text")
#                 chunk_type = hit.entity.get("type") or "content"
#                 if chunk_type != "content" or not text or not text.strip():
#                     continue
#                 candidates.append({
#                     "text": text.strip(),
#                     "access_type": (hit.entity.get("access_type") or "free").lower(),
#                     "url": hit.entity.get("url")
#                 })

#         if not candidates:
#             return {"free": "", "paid": ""}

#         # Re-rank for content
#         pairs = [[query, c["text"]] for c in candidates]
#         scores = self.reranker.predict(pairs)
#         for c, s in zip(candidates, scores):
#             c["score"] = float(s)

#         ranked_chunks = sorted(candidates, key=lambda x: x["score"], reverse=True)
#         top_chunks = ranked_chunks[:final_top_k]

#         grouped = {"free": [], "paid": []}
#         for c in top_chunks:
#             if c["url"]:
#                 grouped[c["access_type"]].append(f"{c['text']}\n(Source: {c['url']})")
#             else:
#                 grouped[c["access_type"]].append(c["text"])

#         return {
#             "free": "\n\n".join(grouped["free"]),
#             "paid": "\n\n".join(grouped["paid"])
#         }
