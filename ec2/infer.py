import sys
import uvicorn
import pandas as pd

from fastapi import FastAPI, HTTPException
from typing import Callable
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util


app_desc = """<h2>Try this app by write work statement with endpoint `/predict`</h2>"""
app = FastAPI(title="NTU Course Recommendation FastAPI", description=app_desc)

# Define model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Get target
df = pd.read_csv('ntu_programmes_course_processed2.csv')
df['all_text'] = df['intro'] + ' ' + df['objective'] + ' ' + df['outline']

target = list(df['all_text'])

@app.get('/index')
async def hello():
	return "hello world"

# Response
class Predictor(BaseModel):
	statement: str = ""

@app.post('/predict')
async def course_rec(input: Predictor):

	# Define empty json
	sorted_pair_json = dict()

	try:
		# Get statement
		query = [input.statement]
		
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

	except:
		e = sys.exc_info()
		raise HTTPException(status_code=500, detail=str(e))
		
	return sorted_pair_json

if __name__ == '__main__':
	uvicorn.run(app, port=8000)