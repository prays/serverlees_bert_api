import os

import uvicorn
import pandas as pd

from fastapi import FastAPI, HTTPException
from typing import Callable
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

MODEL_DIR = os.getenv("MODEL_DIR")

model = SentenceTransformer(MODEL_DIR + '/model_bert')

# Get target
df = pd.read_csv(MODEL_DIR + '/ntu_programmes_course_processed2.csv')
df['all_text'] = df['intro'] + ' ' + df['objective'] + ' ' + df['outline']

target = list(df['all_text'])

def lambda_handler(event, context):
	data = {}
	query = event['statement']

	# Define empty json
	sorted_pair_json = dict()

	# Compute embedding for both lists
	embeddings1 = model.encode(query)
	embeddings2 = model.encode(target)

	# Compute cosine-similarits
	cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

	# Get sorted pair of title course
	pair_similarity = zip(df['title'], cosine_scores.tolist()[0])
	list_pair = list(pair_similarity)
	sorted_pair = sorted(list_pair, key = lambda x: x[1], reverse=True)

	# Get 10 list recommendation and convert to json
	for pair in sorted_pair[:10]:
		sorted_pair_json[pair[0]] = pair[1]
  
	return sorted_pair_json
