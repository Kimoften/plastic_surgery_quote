import dlib
import cv2

# 거리 함수
def distance(x1, y1, x2, y2):
    d = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    return d

def two_image_landmark_cal(img):

    image_path = img

    # 얼굴 감지기와 facial landmark predictor 생성
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/Users/rabbi/.vscode/extrafiles/shape_predictor_68_face_landmarks.dat")  # 미리 훈련된 모델 사용

    # 이미지 읽기
    image = cv2.imread(image_path)
    img_color = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    faces = detector(img_color)
    for face in faces:
        landmarks = predictor(img_color, face)

        # facial landmark 좌표 출력
        landmark_factor_x = []
        landmark_factor_y = []

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_factor_x.append(float(x))
            landmark_factor_y.append(y)

    x, y = landmark_factor_x, landmark_factor_y

    f = []

    # 기준 잡기
    standard = distance(x[0], y[0], x[16], y[16])

    # 왼쪽눈 가로 길이
    eye_left_width = distance(x[39], y[39], x[36], y[36])
    f.append(eye_left_width)

    # 오른쪽눈 가로 길이
    eye_right_width = distance(x[45], y[45], x[42], y[42])
    f.append(eye_right_width)

    # 왼쪽눈 세로 길이
    eye_left_height = distance(x[41], y[41], x[37], y[37]) + distance(x[40], y[40], x[38], y[38])
    f.append(eye_left_height)

    # 오른쪽눈 세로 길이
    eye_right_height = distance(x[47], y[47], x[43], y[43]) + distance(x[46], y[46], x[44], y[44])
    f.append(eye_right_height)

    # 미간 길이
    eye_between = distance(x[42], y[42], x[39], y[39])
    f.append(eye_between)

    # 코 가로 길이(nasal bridge)
    nose_width = distance(x[31], y[31], x[35], y[35])
    f.append(nose_width)

    # 인중
    philtrum_length = distance(x[51], y[51], x[33], y[33])
    f.append(philtrum_length)

    # 입술 가로 길이
    lip_width = distance(x[54], y[54], x[48], y[48])
    f.append(lip_width)

    # 윗입술 두께
    lip_up_thick = distance(x[61], y[61], x[50], y[50]) + distance(x[63], y[63], x[52], y[52])
    f.append(lip_up_thick)

    # 아랫입술 두께
    lip_down_thick = distance(x[66], y[66], x[57], y[57])
    f.append(lip_down_thick)

    # 턱 길이
    chin_length = distance(x[57], y[57], x[8], y[8])
    f.append(chin_length)

    landmark_rate = []

    for rate in f:
        r = rate / standard
        landmark_rate.append(r)

    return landmark_rate