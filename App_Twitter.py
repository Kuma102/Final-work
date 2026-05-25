import streamlit as st
import requests
import pandas as pd
import os

st.title("Итоговая работа. Twitter")

API_URL = "http://127.0.0.1:8000/predict"
DATASET_PATH = "final_extended_dataset.csv"

choice = st.sidebar.selectbox("Выберите раздел:", ["Анализ текста", "Статистика датасета", "Справка"])

if choice == "Анализ текста":
    st.header("1. Проверка комментария")

    user_input = st.text_input("Введите текст отзыва:")

    if st.button("Определить тональность"):
        if not user_input.strip():
            st.warning("Поле пустое")
        else:
            try:
                response = requests.post(API_URL, json={"text": user_input}, timeout=5)

                if response.status_code == 200:
                    result = response.json()

                    st.write(f"**Текст:** {result['text']}")
                    st.write(f"**Результат:** {result['sentiment']}")
                    st.write(f"**Класс:** {result['class_label']}")
                    st.write(f"**Вероятность (0-1):** {result['probability']}")

            except:
                st.error("Ошибка! Проверьте, запущен ли uvicorn сервер API.")

elif choice == "Статистика датасета":
    st.header("2. Данные датасета")

    if not os.path.exists(DATASET_PATH):
        st.error("Файл с данными не найден.")
    else:
        df = pd.read_csv(DATASET_PATH)

        total = len(df)
        df['length'] = df['text'].astype(str).apply(len)
        avg_len = df['length'].mean()

        pos_count = len(df[df['tonality'] == 1])
        neg_count = len(df[df['tonality'] == 0])

        st.write(f"**Общее количество строк:** {total}")
        st.write(f"**Средняя длина текста (символы):** {int(avg_len)}")
        st.write(f"**Количество позитивных (класс 1):** {pos_count} ({round(pos_count / total * 100, 1)}%)")
        st.write(f"**Количество негативных (класс 0):** {neg_count} ({round(neg_count / total * 100, 1)}%)")

        st.write("---")

        st.subheader("График баланса классов (0 и 1)")
        counts_df = pd.DataFrame({"Количество": [neg_count, pos_count]}, index=["Класс 0", "Класс 1"])
        st.bar_chart(counts_df)

        st.subheader("График длин комментариев")
        st.line_chart(df['length'].tail(100))

        st.subheader("Пример строк из таблицы:")
        st.dataframe(df[['text', 'tonality']].head(10))

elif choice == "Справка":
    st.header("3. Справка по командам")
    st.text("""
    Доступные эндпоинты разработанного API:

    1. GET http://127.0.0.1:8000/
       Параметры: нет
       Описание: Проверка работы сервера.

    2. POST http://127.0.0.1:8000/predict
       Формат JSON запроса: {"text": "строка"}
       Формат JSON ответа: {
           "text": "строка",
           "sentiment": "Положительный/Отрицательный",
           "class_label": 1/0,
           "probability": число
       }
    """)