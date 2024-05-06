import pinecone
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

# Pinecone API 불러오기
pinecone.init(
     api_key=os.getenv("PINECONE_API_KEY"),
     environment="gcp-starter"
 )

# # Pinecone index 초기화
# index_name = "surgery"

# if index_name in pinecone.list_indexes():
#     pinecone.delete_index(index_name)

# # Pinecone 새로운 index 생성
# pinecone.create_index(name='surgery', dimension=1536)

# OpenAI API, Pinecone index 불러오기
client = OpenAI()
index = pinecone.Index('surgery')

# 벡터 임베딩 정의
def get_embedding(content, model="text-embedding-ada-002"):
    return client.embeddings.create(input=content, model=model).data[0].embedding

df = pd.read_csv("/Users/rabbi/.vscode/KAIROS/kairos_booth/rag/surgery_solution1.csv", index_col=0, encoding='CP949')
# print(df)
# print(df.columns)

for index_csv, rows in df.iterrows():
    # NaN 값 처리
    sort = rows['sort'] if pd.notna(rows['sort']) else 'unknown'
    sex = rows['sex'] if pd.notna(rows['sex']) else 'unknown'
    cost = rows['cost'] if pd.notna(rows['cost']) else '0'
    metadata = {'Q': rows['Q'], 'sort': sort, 'sex': sex, 'cost': cost}

    value = get_embedding(str(rows['Q']), model="text-embedding-ada-002")
    upsert_data = []
    upsert_data.append((str(index_csv), value, metadata))

    index.upsert(vectors=upsert_data, namespace='surgery')