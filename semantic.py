# -*- coding: utf-8 -*-


# !pip install -q sentence-transformers faiss-cpu numpy pandas matplotlib datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple
import time
import re
import torch
import requests
from io import StringIO

abstracts = [
    {
        "id": 1,
        "title": "Deep Learning for Natural Language Processing",
        "abstract": "This paper explores recent advances in deep learning models for natural language processing tasks. We review transformer architectures including BERT, GPT, and T5, and analyze their performance on various benchmarks including question answering, sentiment analysis, and text classification."
    },
    {
        "id": 2,
        "title": "Climate Change Impact on Marine Ecosystems",
        "abstract": "Rising ocean temperatures and acidification are severely impacting coral reefs and marine biodiversity. This study presents data collected over a 10-year period, demonstrating accelerated decline in reef ecosystems and proposing conservation strategies to mitigate further damage."
    },
    {
        "id": 3,
        "title": "Advancements in mRNA Vaccine Technology",
        "abstract": "The development of mRNA vaccines represents a breakthrough in immunization technology. This review discusses the mechanism of action, stability improvements, and clinical efficacy of mRNA platforms, with special attention to their rapid deployment during the COVID-19 pandemic."
    },
    {
        "id": 4,
        "title": "Quantum Computing Algorithms for Optimization Problems",
        "abstract": "Quantum computing offers potential speedups for solving complex optimization problems. This paper presents quantum algorithms for combinatorial optimization and compares their theoretical performance with classical methods on problems including traveling salesman and maximum cut."
    },
    {
        "id": 5,
        "title": "Sustainable Urban Planning Frameworks",
        "abstract": "This research proposes frameworks for sustainable urban development that integrate renewable energy systems, efficient public transportation networks, and green infrastructure. Case studies from five cities demonstrate reductions in carbon emissions and improvements in quality of life metrics."
    },
    {
        "id": 6,
        "title": "Neural Networks for Computer Vision",
        "abstract": "Convolutional neural networks have revolutionized computer vision tasks. This paper examines recent architectural innovations including residual connections, attention mechanisms, and vision transformers, evaluating their performance on image classification, object detection, and segmentation benchmarks."
    },
    {
        "id": 7,
        "title": "Blockchain Applications in Supply Chain Management",
        "abstract": "Blockchain technology enables transparent and secure tracking of goods throughout supply chains. This study analyzes implementations across food, pharmaceutical, and retail industries, quantifying improvements in traceability, reduction in counterfeit products, and enhanced consumer trust."
    },
    {
        "id": 8,
        "title": "Genetic Factors in Autoimmune Disorders",
        "abstract": "This research identifies key genetic markers associated with increased susceptibility to autoimmune conditions. Through genome-wide association studies of 15,000 patients, we identified novel variants that influence immune system regulation and may serve as targets for personalized therapeutic approaches."
    },
    {
        "id": 9,
        "title": "Reinforcement Learning for Robotic Control Systems",
        "abstract": "Deep reinforcement learning enables robots to learn complex manipulation tasks through trial and error. This paper presents a framework that combines model-based planning with policy gradient methods to achieve sample-efficient learning of dexterous manipulation skills."
    },
    {
        "id": 10,
        "title": "Microplastic Pollution in Freshwater Systems",
        "abstract": "This study quantifies microplastic contamination across 30 freshwater lakes and rivers, identifying primary sources and transport mechanisms. Results indicate correlation between population density and contamination levels, with implications for water treatment policies and plastic waste management."
    }
]


data = pd.DataFrame(abstracts)
print(f"Dataset loaded with {len(data)} scientific papers")

# model_name="all-MiniLM-L6-V2"
model_name="bert-base-nli-mean-tokens"
model=SentenceTransformer(model_name)
print(f"Loaded model: {model_name}")

sentences=data['abstract'].tolist()
print(f"Loaded {len(sentences)} documents")

document_embeddings=model.encode(sentences,show_progress_bar=True)
print(f"Generated {len(document_embeddings)} embeddings with dimenstion {document_embeddings.shape[1]}")

dimenstion=document_embeddings.shape[1]
print(f"Dimension of embeddings: {dimenstion}")

index=faiss.IndexFlatL2(dimenstion)
index.add(np.array(document_embeddings).astype('float32'))
print(f"Created Faiss index with {index.ntotal} vectors")

def semantic_search(query:str,top_k:int=3)->List[Dict]:
  query_embedding=model.encode([query])
  distances,indices=index.search(np.array(query_embedding).astype('float32'),top_k)
  results=[]
  for i,idx in enumerate(indices[0]):
    results.append({
        'id':data.iloc[idx]['id'],
        'title':data.iloc[idx]['title'],
        'abstract':data.iloc[idx]['abstract'],
        'similarity_score':1-distances[0][i]/2
    })




    return results
    # print(indices[0])

test_queries = [
    "How do transformers work in natural language processing?",
    "What are the effects of global warming on ocean life?",
    "Tell me about COVID vaccine development",
    "Latest algorithms in quantum computing",
    "How can cities reduce their carbon footprint?"
]

# query = "Where old man standing alone?"


for query in test_queries:
    print("\n" + "="*80)
    print(f"Query: {query}")
    print("="*80)


    results = semantic_search(query, top_k=3)
    print(results)