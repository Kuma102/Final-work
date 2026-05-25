from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

# Инициализируем приложение FastAPI
app = FastAPI()

MODEL_PATH = 'logistic_regression_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise RuntimeError(
        f"Критические файлы {MODEL_PATH} или {VECTORIZER_PATH} не найдены!")

# Загрузка векторизатора и модельки обучения
tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
log_model = joblib.load(MODEL_PATH)

# Описываем структуру входящего запроса
class CommentInput(BaseModel):
    text: str


# Базовый эндпоинт для проверки API
@app.get("/")
def read_root():
    return {"message": "API запустилось!"}


# Эндпоинт для предугадывания тональности текста
@app.post("/predict")
def predict_tonality(comment: CommentInput):
    if not comment.text.strip():
        raise HTTPException(status_code=400, detail="Текст комментария не может быть пустым")

    try:
        # Векторизация полученного текста
        text_vectorized = tfidf_vectorizer.transform([comment.text])

        # Считаем вероятность принадлежности к классам
        probability_positive = float(log_model.predict_proba(text_vectorized)[0][1])

        # Делаем предсказание класса
        prediction_class = int(log_model.predict(text_vectorized)[0])

        # Формирование текстовое название тональности
        sentiment_label = "Положительный" if prediction_class == 1 else "Отрицательный"

        # Отправка JSON-объекта
        return {
            "text": comment.text,
            "sentiment": sentiment_label,
            "class_label": prediction_class,
            "probability": round(probability_positive, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сервера при классификации: {str(e)}")

#  uvicorn Api_Twitter:app --reload