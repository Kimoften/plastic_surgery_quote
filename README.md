# kairos_proto






# 오류 정리

## landmark 관련

### 문제1: Dlib 라이브러리 설치 오류

    pip install Dlib


위 pip 명령어 입력하면 아래와 같은 오류가 발생함


    CMake Error at CMakeLists.txt:5 (message):
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        You must use Visual Studio to build a python extension on windows.  If you
        are getting this error it means you have not installed Visual C++.  Note
        that there are many flavors of Visual Studio, like Visual Studio for C#
        development.  You need to install Visual Studio for C++.


        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    note: This error originates from a subprocess, and is likely not a problem with pip.
    ERROR: Failed building wheel for Dlib
    Failed to build Dlib
    ERROR: Could not build wheels for Dlib, which is required to install pyproject.toml-based projects


### 해결책
파이썬 버전에 따라서 Dlib 라이브러리 설치가 상이하기 때문에 발생하는 문제임


    pip install cmake

하고,

아래 링크 들어가서 각자 사용하고 있는 파이썬 버전에 맞는 파일 설치

<https://github.com/z-mahmud22/Dlib_Windows_Python3.x>

이후 아래 두 개의 명령어 중 한 개 입력

    pip install opencv-contrib-python dlib
    pip install dlib


여기서 아래처럼 오류 또 뜰 수 있음

    Processing c:\user\~~\dlib-19.24.1-cp311-cp311-win_amd64.whl
    ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory: 'C:\\user\\~~\\dlib-19.24.1-cp311-cp311-win_amd64.whl'

오류 뜨는 이유는, 에러에서 출력된 경로에 위에서 설치한 파일이 없기 때문이므로, 해당 경로로 파일 위치를 옮긴 이후 다시 pip 설치해주면 됨

