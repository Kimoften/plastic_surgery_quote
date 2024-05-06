from id_to_answer import conversation2, conversation3


#얼굴티어
def face_tier():

    response3 = conversation3()
    save_message = conversation2()

    #숫자 뽑아내기
    numeric_response = ""
    for char in response3:
        if char.isdigit():
            numeric_response += char

    nr = int(numeric_response)

    #비교
    if nr <= 100000:
        face_tier = "S급"
    elif nr <= 500000:
        face_tier = "A급"
    elif nr <= 1000000:
        face_tier = "B급"
    elif nr <= 3000000:
        face_tier = "C급"
    else:
        face_tier = "F급"
    
    return save_message, face_tier
