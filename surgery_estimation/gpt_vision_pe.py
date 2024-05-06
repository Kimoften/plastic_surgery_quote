from openai import OpenAI
import base64
import requests

client = OpenAI()

# 이미지를 인코딩하는 함수
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 사용자, 합성 이미지의 base64 문자열을 가져옴
base64_image_1 = encode_image('/Users/rabbi/.vscode/KAIROS/kairos_booth/static/img_store/user.jpg')
base64_image_2 = encode_image('/Users/rabbi/.vscode/KAIROS/kairos_booth/static/img_store/synth.jpg')

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {client.api_key}"
}

payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Let's play a game of Spot the Difference! The first person to guess correctly wins."
                },
                {
                    "type": "text",
                    "text": "Look at the shape of first photo face to see if there's a bulge on cheekbones. If cheekbones are sticking out, then print cheekbones: sticking out. If it is not sticking out, print cheekbones: No problem."
                },
                {
                    "type": "text",
                    "text": "Check the first photo for the periorbital wrinkle. Output the results as follows: periorbital wrinkle: present or absent."
                },
                {
                    "type": "text",
                    "text": "Check the first photo for the smile lines. Output the results as follows: smile line: present or absent."
                },
                #{
                #   "type": "text",
                #    "text": "Compare two pictures, and answer me about the difference in the shape of the nose (compare nose bridge, and tip of nose). You should include the words less and more in your answer. Compare the two photos and print about the contrasting features of the first photo only. However, if there is no obvious difference, print no big difference. Output the results as follows: Nose shape: The tip of the nose is sharper (or rounder), the nose bridge is less pointed(or the opposite word)."
                #},
                #{
                #    "type": "text",
                #    "text": "Compare two pictures, and answer me the both shape of the face. There are three types of face shapes. The 'oval face' has ideally proportioned cheeks and chin, with less flesh on the cheeks. 'Rounded faces' have flesh on the face and no visible skeleton around the edges. And 'square faces' have a well-developed jaw skeleton, giving the face shape more of a square. Output the results as follows: Face shape: 'type of face' (first photo) , 'type of face' (second photo)."
                #},
                #{
                #    "type": "text",
                #    "text": "Compare two pictures and Look at the shape of the jaw below the ear and answer me if it's sharper or square. If the jawline below the ear is angled, it's close to SQUARE. You should include the words less and more in your answer. Compare the two photos and print about the contrasting features of the first photo only. However, if there is no obvious difference, print no big difference. Output the results as follows: Jawline: Sharper (or more square shape)."
                #},
                {
                    "type": "text",
                    "text": "Please compare the presence or absence of double eyelid with the second picture. Compare the two photos and print about the contrasting features of the first photo only. Output the results as follows: Double eyelid: If you can't observe it in both photos, output 'No problem'; If double eyelids are observable in the first photo but not in the second, output 'present'. ; if double eyelids can't observe in the first photo and can observe in the second, print 'absent'."
                },
                {
                    "type": "text",
                    "text": "Please don't write a summary paragraph at the end"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image_1}",
                        "detail": "high"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image_2}",
                        "detail": "high"
                    }
                }
            ]
        }
    ],
    "max_tokens": 2000
}

def gpt_vision():
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    gpt_vision_answer = response.json()

    gpt_vision_answers = [item.strip() for choice in gpt_vision_answer["choices"] for item in choice["message"]["content"].split("\n")]

    return gpt_vision_answers