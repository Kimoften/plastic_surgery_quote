import pinecone
from openai import OpenAI
import os
import sys
sys.path.append('/Users/rabbi/.vscode/KAIROS/kairos_booth')
from surgery_estimation.landmark_output import landmark_answer
from surgery_estimation.gpt_vision_pe import gpt_vision
from image_recognition.predict import image_compare
# import app
from dotenv import load_dotenv
load_dotenv()

# OpenAI API 설정
client = OpenAI()
MODEL1 = "text-embedding-ada-002"

# Pinecone API 설정
pinecone_instance = pinecone.Pinecone(api_key="PINECONE_API_KEY")
# pinecone.init(
#     api_key=os.getenv("PINECONE_API_KEY"),
#     environment="gcp-starter"
# )
index = pinecone.Index(index_name='surgery', host='https://surgery-68lt2mc.svc.gcp-starter.pinecone.io', api_key='192581e4-9dcf-4e67-90db-16f39dd3c41f')

# gender = app.gender
# print(gender)

# input_data에서 metadata 뽑기 위해 키워드 추출 후 metadata 만들기
def get_metadata(input_data):
    metadata = {}

    if "eye" in input_data.lower():
        metadata["sort"] = "eye"
    elif "nose" in input_data.lower():
        metadata["sort"] = "nose"
    elif "lip" in input_data.lower():
        metadata["sort"] = "lip"
    elif "chin" in input_data.lower():
        metadata["sort"] = "chin"
    elif "cheek" in input_data.lower():
        metadata["sort"] = "cheek"
    elif "wrinkle" in input_data.lower():
        metadata["sort"] = "wrinkle"
    elif "line" in input_data.lower():
        metadata["sort"] = "line"
    else:
        metadata["sort"] = "unknown"

    if "long" in input_data.lower():
        metadata["feature"] = "long"
    elif "short" in input_data.lower():
        metadata["feature"] = "short"
    elif "present" in input_data.lower():
        metadata["feature"] = "present"
    elif "absent" in input_data.lower():
        metadata["feature"] = "absent"
    else:
        metadata["feature"] = "unknown"

    # if "glabella: shorter" in input_data.lower():
    #     metadata["sex"] = gender[0]
    # elif "glabella: shorter" in input_data.lower():
    #     metadata["sex"] = gender[0]
    # else:
    #     metadata["sex"] = "unknown"
    
    return metadata

# OpenAI를 사용하여 텍스트 임베딩 생성
def embed_text(text):
    result = client.embeddings.create(model=MODEL1, input=text)
    return result.data[0].embedding

# Pinecone을 사용하여 벡터 유사도 검색
def search_pinecone(embeddings, metadata, top_k=1):
    search_results = index.query(
        vector=embeddings,
        filter=metadata,
        top_k=top_k,
        namespace='surgery',
        include_metadata=True)
    return search_results

# 견적 결과 리스트
# vision = gpt_vision()
# landmark = landmark_answer()
# image_detect = image_compare()
# input_list = landmark + image_detect



def output_id():
    output_id_lists = []
    # 리스트의 각 항목을 임베딩하여 Pinecone에서 검색

    from surgery_estimation.landmark_output import landmark_answer
    from image_recognition.predict import image_compare

    landmark = landmark_answer()
    image_detect = image_compare()
    input_list = landmark + image_detect

    for item in input_list:
        meta = get_metadata(item)
        # print(meta)
        item_vec = embed_text(item)
        search_results = search_pinecone(item_vec, meta, top_k=1)

        for match in search_results['matches']:
            output_id_lists.append(int(match['id']))
    return output_id_lists

# print(output_id_list)
        
# def rag_cleanup():
#     output_id_list.clear