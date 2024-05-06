from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import threading
import time
from dotenv import load_dotenv
import subprocess
import requests
import shutil
from urllib.parse import urlparse, unquote

dotenv_path = '/Users/rabbi/.vscode/KAIROS/kairos_booth/rag/.env'
load_dotenv(dotenv_path)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')
CORS(app, resources={r"/submit": {"origins": "http://127.0.0.1:5000"}})

#gender = []

template_dir = os.path.abspath('/Users/rabbi/.vscode/KAIROS/kairos_booth/templates')
app.template_folder = template_dir

def download_file(url, local_path):
    response = requests.get(url, stream=True)
    with open(local_path, 'wb') as file:
        shutil.copyfileobj(response.raw, file)
    del response

def copy_file(src_path, dest_path):
    try:
        # 파일 다운로드
        download_file(src_path, dest_path)
        print(f'파일이 성공적으로 복사되었습니다. ({src_path} -> {dest_path})')
    except Exception as e:
        print(f'파일 복사 중 오류가 발생했습니다: {str(e)}')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/take_my_picture.html')
def index1():
        return render_template('take_my_picture.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    try:
        # 클라이언트로부터 이미지 파일을 받음
        image1 = request.files['image']

        # 이미지를 저장할 경로 설정 (예시: 현재 디렉토리의 static 폴더 안에 저장)
        # save_path = 'static/uploaded_images/img_store' + image1.filename
        # image1.save(save_path)

        upload_folder = '/Users/rabbi/.vscode/KAIROS/kairos_booth/static/img_store'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        filename1 = secure_filename("user.jpg")
        image1.save(os.path.join(upload_folder, filename1))

        # 성공 응답 반환
        return jsonify(success=True, message='Image upload successful')

    except Exception as e:
        # 실패 응답 반환
        return jsonify(success=False, error=str(e))

# if __name__ == '__main__':
#     app.run(debug=True)

@app.route('/select_wanttobe_face.html')
def index2():
        return render_template('select_wanttobe_face.html')
    
@app.route('/submit', methods=['POST', 'GET'])
def submit():
    try:
        # print("Request를 받았습니다...")
        if 'image2_url' not in request.form:
            # print("이미지2가 request.files에 있음")
            return jsonify({'success': False, 'error': 'No image file part'})


        # image2 = request.files['image2']

        # base_dir = '/Users/rabbi/.vscode/KAIROS/kairos_booth'

        image2_url = request.form['image2_url']
        # parsed_url = urlparse(image2_url)

        # image2_url_absolute = os.path.join(base_dir, unquote(parsed_url.path))


        # src_file = image2_url
        dest_file = '/Users/rabbi/.vscode/KAIROS/kairos_booth/static/img_store/want.jpg'

        copy_file(image2_url, dest_file)

        # # image1.save(os.path.join(upload_folder, filename1))
        # image2.save(os.path.join(upload_folder, filename2))

        image1 = '/Users/rabbi/.vscode/KAIROS/kairos_booth/static/img_store/user.jpg'
        image2 = dest_file

        from image_synthesis.synth_image import synth_process
        synth_process()

        if image1 and image2:
            print("Images received successfully.")
            from rag.id_to_answer import conversation
            response = conversation()
            session['response'] = response  # 세션에 결과 저장
            return redirect('/result')
            # return jsonify({'success': True})
        
    except Exception as e:
        session['response'] = str(e)  # 오류 메시지 저장
        return jsonify({'success': False, 'error': str(e)})

# def download_and_save_image(url, filename):
#     try:
#         response = requests.get(url, stream=True)
#         response.raise_for_status()

#         upload_folder = '/Users/rabbi/.vscode/KAIROS/kairos_booth/static/img_store'
#         if not os.path.exists(upload_folder):
#             os.makedirs(upload_folder)

#         file_path = os.path.join(upload_folder, secure_filename(filename))

#         with open(file_path, 'wb') as file:
#             for chunk in response.iter_content(chunk_size=8192):
#                 file.write(chunk)

#         return file_path

#     except Exception as e:
#         print(f"Error downloading and saving image: {e}")
#         return None


def delayed_cleanup():
    time.sleep(1)  # Delay to ensure response is sent
    try:
        os.remove('/Users/rabbi/.vscode/KAIROS/kairos_booth/static/img_store/user.jpg')
        os.remove('/Users/rabbi/.vscode/KAIROS/kairos_booth/static/img_store/want.jpg')
        
    except Exception as e:
        print(f"Error during file cleanup: {e}")

def cleanup():
    time.sleep(1)
    session.pop('response', None)

@app.route('/result')
def result():
    response = session.get('response', 'No response')
    # threading.Thread(target=delayed_cleanup).start()
    # threading.Thread(target=cleanup).start()

    # from surgery_estimation import landmark_output

    # print(landmark_output.landmark_answer)

    return render_template('result.html', response=response)

def delayed_cleanup2():
    time.sleep(1)
    try:
        os.remove('/Users/rabbi/.vscode/KAIROS/kairos_booth/static/img_store/synth.jpg')
    except Exception as e:
        print(f"Error during file cleanup: {e}")

@app.route('/redo')
def redo():
    try:
        session.pop('response', None)  # 세션에서 'response'를 제거
        delayed_cleanup()
        delayed_cleanup2()
        # from rag.id_to_answer import rag_cleanup
        # rag_cleanup()  # delayed_cleanup 함수를 호출
        # from rag.id_to_answer import conversation2
        # conversation2()
        return render_template('index.html')  # 메인 페이지로 리디렉션
    except Exception as e:
        print(f"Error occurred: {e}")

@app.route('/save_result')
def save_result():
    save_message = session.get('save_message', 'No message')
    save_face_tier = session.get('save_face_tier', 'No tier')

    from rag.card import face_tier

    save_message, save_face_tier = face_tier()
    # card.py 스크립트 실행
    try:
        subprocess.run(['python', 'rag/card.py'], check=True)
    except subprocess.CalledProcessError as e:
        response = f"card.py 실행 중 오류 발생: {e}"

    threading.Thread(target=delayed_cleanup2).start()

    return render_template('save_result.html', save_message=save_message, save_face_tier=save_face_tier)

if __name__ == '__main__':
    app.run(debug=True)