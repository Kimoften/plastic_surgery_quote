import onnx
import onnxruntime
import cv2
import torch
import torchvision.transforms as transforms

# 이미지 전처리 함수
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 이미지를 BGR 형식으로 읽으므로 RGB로 변환
    image = transforms.ToPILImage()(image)  # OpenCV 이미지를 PIL 이미지로 변환
    preprocess = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)  # 배치 차원 추가
    return image

# ONNX 모델 경로
onnx_model_path = "/Users/rabbi/.vscode/KAIROS/kairos_booth/image_recognition/ImageClassifier.onnx"

# ONNX 모델 로드
onnx_model = onnx.load(onnx_model_path)

# ONNX 런타임 생성
ort_session = onnxruntime.InferenceSession(onnx_model_path)

def image_recog(img):

    # 이미지 경로
    image_path = img

    # 이미지 전처리하기
    input_image = preprocess_image(image_path)

    # ONNX 모델 입력에 전처리된 이미지 넣기
    ort_inputs = {ort_session.get_inputs()[0].name: input_image.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    # print("ONNX 모델의 예측 결과:", ort_outputs)

    # 예측 결과에서 최댓값 인덱스 찾기
    predicted_class_index = torch.argmax(torch.Tensor(ort_outputs[0])).item()

    classes = ['Sharp','Square']

    # 예측된 라벨 출력
    predicted_label = classes[predicted_class_index]
    output = []
    str = "Jawline: "+ predicted_label
    output.append(str)
    return output

clear = []

def image_compare():
    user_jaw = image_recog('/Users/rabbi/.vscode/KAIROS/kairos_booth/static/img_store/user.jpg')
    synth_jaw = image_recog('/Users/rabbi/.vscode/KAIROS/kairos_booth/static/img_store/synth.jpg')

    if user_jaw != synth_jaw:
        return user_jaw
    else:
        return clear