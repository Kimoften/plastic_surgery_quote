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



import pandas as pd
# import vector_search
from openai import OpenAI
from dotenv import load_dotenv
import time
load_dotenv()

client = OpenAI()
MODEL2 = "gpt-3.5-turbo-1106"

# 엑셀 파일 불러오기   
df = pd.read_csv("/Users/rabbi/.vscode/KAIROS/kairos_booth/rag/surgery_solution1.csv", index_col=0, encoding='CP949')
# print(df)



# def rag_cleanup():
#     output_id_list.clear

# def solution():
#     solution_answer = []

#     for index_file, rows in df.iterrows():
#         for id in output_id_list:
#             if id == index_file:
#                 solution_answer.append("현재 내 얼굴의 문제점: " + rows['Q'] + "\r\n" + rows['A'] + "\r\n비용: " + rows['cost'])
#             else:
#                 continue
# print(solution_answer)

def reset_messages():
    # 초기화할 메시지
    messages = [
        {"role": "system", "content": "Please follow the instructions below and ANSWER IN KOREAN.\
                [Role]\
                You are a plastic surgeon, and user is a patient who has come for a consultation.\
                [Answer requirements]\
                1. refer to the user's question and explain plastic surgery estimate and cost.\
                2. Your answer should include the following content:\
                a. Tell user what the user's current facial problem is. The user doesn't know what the problem is, so answer in a diagnostic tone.\
                b-1. If the cost is zero, answer the user's question in a humorous tone.\
                B-2. If surgery is required, answer the name of the surgery and the details of the surgery in a professional tone, and answer the side effects in a humorous tone.\
                3. At the beginning of your answer, provide the total cost of the user's plastic surgery and a one-line comment about the user as a plastic surgeon.\
                4. At the end of your answer, conclude your diagnosis with a random comment (encouragement/teasing, etc.) that you can give the user about the above.\
                5. Do not include any of the '(facial area): (condition)' wordings from the user's question in your answer (e.g. glabella: longer, etc.).\
                6. When sending a response, classify each type of surgery into paragraphs and print each paragraph with a single space between them.\
                a. When separating paragraphs, put line breaks between paragraphs."}]
    
    return messages

# GPT-4 답변 출력
def conversation():
    solution_answer = []

    # from vector_search import output_id
    output_id_list = output_id()

    for index_file, rows in df.iterrows():
        for id in output_id_list:
            if id == index_file:
                solution_answer.append("현재 내 얼굴의 문제점: " + rows['Q'] + "\r\n" + rows['A'] + "\r\n비용: " + rows['cost'])
            else:
                continue
    
    # messages = [{"role": "system", "content": "Please follow the instructions below and ANSWER IN KOREAN.\
    #             [Role]\
    #             You are a plastic surgeon, and user is a patient who has come for a consultation.\
    #             [Answer requirements]\
    #             1. refer to the user's question and explain plastic surgery estimate and cost.\
    #             2. Your answer should include the following content:\
    #             a. Tell user what the user's current facial problem is. The user doesn't know what the problem is, so answer in a diagnostic tone.\
    #             b-1. If the cost is zero, answer the user's question in a humorous tone.\
    #             B-2. If surgery is required, answer the name of the surgery and the details of the surgery in a professional tone, and answer the side effects in a humorous tone.\
    #             3. At the beginning of your answer, provide the total cost of the user's plastic surgery and a one-line comment about the user as a plastic surgeon.\
    #             4. At the end of your answer, conclude your diagnosis with a random comment (encouragement/teasing, etc.) that you can give the user about the above.\
    #             5. Do not include any of the '(facial area): (condition)' wordings from the user's question in your answer (e.g. glabella: longer, etc.).\
    #             6. When sending a response, classify each type of surgery into paragraphs and print each paragraph with a single space between them.\
    #             a. When separating paragraphs, put line breaks between paragraphs."}]
            
    messages = reset_messages()

    messages.append({"role": "user", "content": f"{solution_answer}"})
    
    completion = client.chat.completions.create(model=MODEL2, messages=messages, temperature=0, max_tokens=1000)
    
    # print(completion)
    response = completion.choices[0].message.content
    # messages.append({"role": "assistant", "content": response})

    return response
    # print(f"당신만의 성형외과 의사: {response}")

# def rag_cleanup():
#     solution_answer = None
    

# conversation()

# def conversation2():
#     reset_answer = []

#     messages2 = [{"role": "system", "content": "reset."}]
    
    
#     messages2.append({"role": "user", "content": f"{reset_answer}"})

#     completion2 = client.chat.completions.create(model=MODEL2, messages=messages2, temperature=0, max_tokens=1000)

#     response2 = completion2.choices[0].message.content

#     return response2


# def conversation3():
#     messages3 = [{"role": "system", "content": "Please follow the instructions below and ANSWER IN KOREAN.\
#                   [Answer requirements]\
#                   1. refer to the user's question and return the total cost of surgery prices:\
#                   2. 답변의 형태는 '총비용: (단위: 원)'으로 고정해줘."}]
    
    
#     messages3.append({"role": "user", "content": f"{solution_answer}"})

#     completion3 = client.chat.completions.create(model=MODEL2, messages=messages3, temperature=0, max_tokens=1000)

#     response3 = completion3.choices[0].message.content

#     return response3

if __name__ == "__main__":
    conversation()