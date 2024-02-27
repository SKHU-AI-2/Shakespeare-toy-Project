from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
import numpy as np

app = FastAPI()

# 모델 로드
model_path = 'shakespeare_text_gen_model.h5'
model = load_model(model_path)

# 고유 문자 및 매핑 초기화
chars = ['\n', ' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_to_index = dict((c, i) for i, c in enumerate(chars))
index_to_char = dict((i, c) for i, c in enumerate(chars))

# 정적 파일과 템플릿 디렉토리 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 텍스트 생성 함수
def generate_custom_text(length, diversity):
    start_index = np.random.randint(0, len(chars) - 1)
    generated = ''
    sentence = [np.random.choice(chars) for _ in range(40)]
    generated += ''.join(sentence)
    for i in range(length):
        x_pred = np.zeros((1, 40, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_index[char]] = 1.
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.random.choice(len(chars), p=preds)
        next_char = index_to_char[next_index]
        generated += next_char
        sentence = sentence[1:] + [next_char]
    return generated

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate/")
async def generate_text_endpoint(request: Request, length: int = Form(...), diversity: float = Form(...)):
    generated_text = generate_custom_text(length, diversity)
    return templates.TemplateResponse("index.html", {"request": request, "generated_text": generated_text})