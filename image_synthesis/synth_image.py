import base64
from PIL import Image
import io
import requests

from typing import List, Tuple
from PIL import Image
from pydantic import BaseModel, Field
from enum import Enum
import base64, io
from io import BytesIO
from typing import List, Tuple, Optional
import numpy as np
import requests
import safetensors


class InpaintingWhen(Enum):
    NEVER = "Never"
    BEFORE_UPSCALING = "Before Upscaling/all"
    BEFORE_RESTORE_FACE = "After Upscaling/Before Restore Face"
    AFTER_ALL = "After All"


class InpaintingOptions(BaseModel):
    inpainting_denoising_strengh: float = Field(
        description="Inpainting denoising strenght", default=0, lt=1, ge=0
    )
    inpainting_prompt: str = Field(
        description="Inpainting denoising strenght",
        examples=["Portrait of a [gender]"],
        default="Portrait of a [gender]",
    )
    inpainting_negative_prompt: str = Field(
        description="Inpainting denoising strenght",
        examples=[
            "Deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation"
        ],
        default="",
    )
    inpainting_steps: int = Field(
        description="Inpainting steps",
        examples=["Portrait of a [gender]"],
        ge=1,
        le=150,
        default=20,
    )
    inpainting_sampler: str = Field(
        description="Inpainting sampler", examples=["Euler"], default="Euler"
    )
    inpainting_model: str = Field(
        description="Inpainting model", examples=["Current"], default="Current"
    )
    inpainting_seed: int = Field(description="Inpainting Seed", ge=-1, default=-1)


class InswappperOptions(BaseModel):
    face_restorer_name: str = Field(
        description="face restorer name", default="CodeFormer"
    )
    restorer_visibility: float = Field(
        description="face restorer visibility", default=1, le=1, ge=0
    )
    codeformer_weight: float = Field(
        description="face restorer codeformer weight", default=1, le=1, ge=0
    )
    upscaler_name: str = Field(description="upscaler name", default=None)
    improved_mask: bool = Field(description="Use Improved Mask", default=False)
    color_corrections: bool = Field(description="Use Color Correction", default=False)
    sharpen: bool = Field(description="Sharpen Image", default=False)
    erosion_factor: float = Field(description="Erosion Factor", default=1, le=10, ge=0)


class FaceSwapUnit(BaseModel):
    # The image given in reference
    source_img: str = Field(
        description="base64 reference image",
        examples=["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQECWAJYAAD...."],
        default=None,
    )
    # The checkpoint file
    source_face: str = Field(
        description="face checkpoint (from models/faceswaplab/faces)",
        examples=["my_face.safetensors"],
        default=None,
    )
    # base64 batch source images
    batch_images: Tuple[str] = Field(
        description="list of base64 batch source images",
        examples=[
            "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQECWAJYAAD....",
            "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQECWAJYAAD....",
        ],
        default=None,
    )

    # Will blend faces if True
    blend_faces: bool = Field(description="Will blend faces if True", default=True)

    # Use same gender filtering
    same_gender: bool = Field(description="Use same gender filtering", default=False)

    # Use same gender filtering
    sort_by_size: bool = Field(description="Sort Faces by size", default=False)

    # If True, discard images with low similarity
    check_similarity: bool = Field(
        description="If True, discard images with low similarity", default=False
    )
    # if True will compute similarity and add it to the image info
    compute_similarity: bool = Field(
        description="If True will compute similarity and add it to the image info",
        default=False,
    )

    # Minimum similarity against the used face (reference, batch or checkpoint)
    min_sim: float = Field(
        description="Minimum similarity against the used face (reference, batch or checkpoint)",
        default=0.0,
    )
    # Minimum similarity against the reference (reference or checkpoint if checkpoint is given)
    min_ref_sim: float = Field(
        description="Minimum similarity against the reference (reference or checkpoint if checkpoint is given)",
        default=0.0,
    )

    # The face index to use for swapping
    faces_index: Tuple[int] = Field(
        description="The face index to use for swapping, list of face numbers starting from 0",
        default=(0,),
    )

    reference_face_index: int = Field(
        description="The face index to use to extract face from reference",
        default=0,
    )

    pre_inpainting: Optional[InpaintingOptions] = Field(
        description="Inpainting options",
        default=None,
    )

    swapping_options: Optional[InswappperOptions] = Field(
        description="PostProcessing & Mask options",
        default=None,
    )

    post_inpainting: Optional[InpaintingOptions] = Field(
        description="Inpainting options",
        default=None,
    )

    def get_batch_images(self) -> List[Image.Image]:
        images = []
        if self.batch_images:
            for img in self.batch_images:
                images.append(base64_to_pil(img))
        return images


class PostProcessingOptions(BaseModel):
    face_restorer_name: str = Field(description="face restorer name", default=None)
    restorer_visibility: float = Field(
        description="face restorer visibility", default=1, le=1, ge=0
    )
    codeformer_weight: float = Field(
        description="face restorer codeformer weight", default=1, le=1, ge=0
    )

    upscaler_name: str = Field(description="upscaler name", default=None)
    scale: float = Field(description="upscaling scale", default=1, le=10, ge=0)
    upscaler_visibility: float = Field(
        description="upscaler visibility", default=1, le=1, ge=0
    )
    inpainting_when: InpaintingWhen = Field(
        description="When inpainting happens",
        examples=[e.value for e in InpaintingWhen.__members__.values()],
        default=InpaintingWhen.NEVER,
    )

    inpainting_options: Optional[InpaintingOptions] = Field(
        description="Inpainting options",
        default=None,
    )


class FaceSwapRequest(BaseModel):
    image: str = Field(
        description="base64 reference image",
        examples=["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQECWAJYAAD...."],
        default=None,
    )
    units: List[FaceSwapUnit]
    postprocessing: Optional[PostProcessingOptions] = None


class FaceSwapResponse(BaseModel):
    images: List[str] = Field(description="base64 swapped image", default=None)
    infos: Optional[List[str]]  # not really used atm

    @property
    def pil_images(self) -> Image.Image:
        return [base64_to_pil(img) for img in self.images]


class FaceSwapCompareRequest(BaseModel):
    image1: str = Field(
        description="base64 reference image",
        examples=["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQECWAJYAAD...."],
        default=None,
    )
    image2: str = Field(
        description="base64 reference image",
        examples=["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQECWAJYAAD...."],
        default=None,
    )


class FaceSwapExtractRequest(BaseModel):
    images: List[str] = Field(
        description="base64 reference image",
        examples=["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQECWAJYAAD...."],
        default=None,
    )
    postprocessing: Optional[PostProcessingOptions]


class FaceSwapExtractResponse(BaseModel):
    images: List[str] = Field(description="base64 face images", default=None)

    @property
    def pil_images(self) -> Image.Image:
        return [base64_to_pil(img) for img in self.images]


def pil_to_base64(img: Image.Image) -> np.array:  # type:ignore
    if isinstance(img, str):
        img = Image.open(img)

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_data = buffer.getvalue()
    base64_data = base64.b64encode(img_data)
    return base64_data.decode("utf-8")


def base64_to_pil(base64str: Optional[str]) -> Optional[Image.Image]:
    if base64str is None:
        return None
    if "base64," in base64str:  # check if the base64 string has a data URL scheme
        base64_data = base64str.split("base64,")[-1]
        img_bytes = base64.b64decode(base64_data)
    else:
        # if no data URL scheme, just decode
        img_bytes = base64.b64decode(base64str)
    return Image.open(io.BytesIO(img_bytes))


def compare_faces(
    image1: Image.Image, image2: Image.Image, base_url: str = "http://localhost:7860"
) -> float:
    request = FaceSwapCompareRequest(
        image1=pil_to_base64(image1),
        image2=pil_to_base64(image2),
    )

    result = requests.post(
        url=f"{base_url}/faceswaplab/compare",
        data=request.json(),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )

    return float(result.text)


def safetensors_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as file:
        file_bytes = file.read()
        return "data:application/face;base64," + base64.b64encode(file_bytes).decode(
            "utf-8"
        )


def base64_to_safetensors(base64str: str, output_path: str) -> None:
    try:
        base64_data = base64str.split("base64,")[-1]
        file_bytes = base64.b64decode(base64_data)
        with open(output_path, "wb") as file:
            file.write(file_bytes)
        with safetensors.safe_open(output_path, framework="pt") as f:
            print(output_path, "keys =", f.keys())
    except Exception as e:
        print("Error : failed to convert base64 string to safetensor", e)
        import traceback

        traceback.print_exc()

address = 'http://127.0.0.1:7860'

# Function to load an image from file path and convert it to base64

def load_and_convert_to_base64(image_path):
    try:
        # 파일에서 이미지 컨텐츠 읽기
        with open(image_path, 'rb') as image_file:
            # 이미지 컨텐츠를 base64로 변환
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_image
    except Exception as e:
        print(f"이미지 로드 또는 변환 중 오류 발생: {str(e)}")
        return None

# Paths to source images
man_image_path = '/Users/rabbi/.vscode/KAIROS/kairos_booth/static/img_store/user.jpg'
man_image2_path = '/Users/rabbi/.vscode/KAIROS/kairos_booth/static/img_store/want.jpg'
test_image_path = '/Users/rabbi/.vscode/KAIROS/kairos_booth/static/img_store/user.jpg'


def synth_process():
    # Face unit :
    unit1 = FaceSwapUnit(
        source_img=load_and_convert_to_base64(man_image_path),  # Load and convert to base64
        faces_index=(0,)  # Replace first face
    )

    unit2 = FaceSwapUnit(
        source_img=load_and_convert_to_base64(man_image2_path),  # Load and convert to base64
        faces_index=(0,)  # Replace first face
    )

    # Post-processing config :
    pp = PostProcessingOptions(
        face_restorer_name="CodeFormer",
        codeformer_weight=0.7,
        restorer_visibility=0.7
    )

    # Prepare the request
    request = FaceSwapRequest(
        image=load_and_convert_to_base64(test_image_path),  # Load and convert to base64
        units=[unit1,unit2],
        postprocessing=pp
    )

    result = requests.post(url=f'{address}/faceswaplab/swap_face', data=request.model_dump_json(), headers={"Content-Type": "application/json; charset=utf-8"})
    response = FaceSwapResponse.model_validate(result.json())

    # Extract image URL from the response
    #image_url = result.json()["images"][0]  # 이미지 URL은 리스트 형태로 반환될 수 있으므로 주의

    # Print the image URL
    #print(f"Swapped Image URL: {image_url}")

    for img in response.pil_images:
        image3 = img.save("/Users/rabbi/.vscode/KAIROS/kairos_booth/static/img_store/synth.jpg", "JPEG", quality = 100)
        # img.show()
synth_process()