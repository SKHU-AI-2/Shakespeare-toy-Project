from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
import numpy as np
import os

app = FastAPI()

# 모델 및 데이터 초기화
model_path = 'shakespeare_text_gen_model.h5'
text_path = 'shakespeare.txt'
model = load_model(model_path)
text = open(text_path, 'rb').read().decode('utf-8')
chars = sorted(set(text))
char_to_index = dict((c, i) for i, c in enumerate(chars))
index_to_char = dict((i, c) for i, c in enumerate(chars))

# 정적 파일과 템플릿 디렉토리 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 텍스트 생성 함수
def generate_custom_text(length, diversity):
    start_index = np.random.randint(0, len(text) - 40 - 1)
    generated = ''
    sentence = text[start_index: start_index + 40]
    generated += sentence
    for i in range(length):
        x_pred = np.zeros((1, 40, len(chars)), dtype=np.bool_)
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_index[char]] = 1.
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.random.choice(len(chars), p=preds)
        next_char = index_to_char[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
    return generated

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate/")
async def generate_text_endpoint(request: Request, length: int = Form(...), diversity: float = Form(...)):
    generated_text = generate_custom_text(length, diversity)
    return templates.TemplateResponse("index.html", {"request": request, "generated_text": generated_text})