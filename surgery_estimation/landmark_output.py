from surgery_estimation.landmark_output_cal import two_image_landmark_cal
import numpy as np

def landmark_answer():
    result_a = two_image_landmark_cal('/Users/rabbi/.vscode/KAIROS/kairos_booth/static/img_store/user.jpg')
    result_b = two_image_landmark_cal('/Users/rabbi/.vscode/KAIROS/kairos_booth/static/img_store/synth.jpg')
    # print(result_a, result_b)

    r = np.array(result_a) / np.array(result_b)
    # print(r)

    # 요소 추가 시 수정
    output_list = ["Left eye width", "Right eye width", "Left eye height", "Right eye height", "Glabella", "Nose width", "Philtrum", "Lip width", "Upper lip thickness", "Lower lip thickness", "Chin length"]

    landmark_answers = []

    n=0

    for i in r:
        if n == 10:
            if 1.0526315789 <= i:
                landmark_answers.append(output_list[n] + ": longer")
            elif 0.93 < i <= 0.95:
                landmark_answers.append(output_list[n] + ": shorter")
            elif i <= 0.93:
                landmark_answers.append(output_list[n] + ": absent")
            else:
                n += 1
                continue
        else:
            if 1.0526315789 <= i:
                landmark_answers.append(output_list[n] + ": longer")
            elif i <= 0.95:
                landmark_answers.append(output_list[n] + ": shorter")
            else:
                n += 1
                continue
        n += 1

    return landmark_answers
