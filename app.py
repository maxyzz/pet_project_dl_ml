import os
import pickle
from datetime import datetime
from typing import Any, Dict, List
import hashlib

import numpy as np
import pandas as pd
from fastapi import FastAPI
from loguru import logger

from database import postgres_connection
from schema import PostGet, Response

def get_model_path(path: str, type: str) -> str:
    """
    Возвращает полный путь к файлу модели с учетом переменной MODELS_DIR в LMS.
    """
    if os.environ.get("IS_LMS") == "1":
        if type=='control':
            path = 'model_control.pkl'
        elif type=='test':
            path = 'model_test.pkl'
        models_dir = os.environ.get("MODELS_DIR", "")
        if models_dir:
            return os.path.join(models_dir, path)
    return path

SALT = "exp_my_salt"
CONTROL_PERCENT = 50 

def get_exp_group(user_id: int) -> str:
    hash_input = f"{user_id}_{SALT}".encode()
    hash_int = int(hashlib.md5(hash_input).hexdigest(), 16)
    bucket = hash_int % 100
    
    if bucket < CONTROL_PERCENT:
        return "control"
    return "test"

# === Вспомогательные функции ===
def load_sql(query: str, dtypes: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Выполняет SQL-запрос через соединение postgres_connection и возвращает DataFrame.

    Аргументы:
        query: SQL-запрос
        dtypes: словарь типов колонок для pd.read_sql (по умолчанию None)

    Возвращает:
        pd.DataFrame с результатом запроса

    Исключения:
        RuntimeError, если произошла ошибка при выполнении запроса
    """
    conn = postgres_connection()

    try:
        df = pd.read_sql(query, conn, dtype=dtypes)
    except Exception as e:
        raise RuntimeError(
            f"❌ Ошибка при выполнении SQL-запроса: {e}\nЗапрос: {query}"
        ) from e
    finally:
        conn.close()

    return df


def load_model(model_path: str, type: str):
    actual_path = get_model_path(model_path,type)
    logger.info(f"Загрузка модели из файла {actual_path}...")

    try:
        with open(actual_path, "rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        models_dir = os.environ.get("MODELS_DIR", ".")
        files_in_dir = os.listdir(models_dir) if os.path.exists(models_dir) else []
        raise FileNotFoundError(
            f"❌ Файл модели не найден: {actual_path}\n"
            f"Файлы в папке моделей ({models_dir}): {files_in_dir}"
        )
    except Exception as e:
        raise RuntimeError(f"❌ Ошибка при загрузке модели: {e}") from e

    logger.success(f"Модель {actual_path} успешно загружена")
    return model


# === Загрузка основных ресурсов ===
logger.info("Инициализация сервиса...")

# Создаём объект FastAPI
app = FastAPI()

# Загружаем модель в память
model_control = load_model(model_path='pipe_hgb.pkl', type = 'control')
model_test = load_model(model_path='pipe_hgb_dl.pkl', type = 'test')


YOUR_LOGIN = 'maksim-zatsepin-eyb5957'  

user_features_test = load_sql(query=f'select * from public."{YOUR_LOGIN}_user_features_dl"')

post_features_test = load_sql(query=f'select * from public."{YOUR_LOGIN}_post_features_dl"')

user_features_control = load_sql(query=f'select * from public."{YOUR_LOGIN}_user_features"')

post_features_control = load_sql(query=f'select * from public."{YOUR_LOGIN}_post_features"')

logger.success("Сервис успешно инициализирован")

# Эндпойнт для получения рекомендаций
@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(user_id: int, dt: datetime, limit: int = 10) -> Response:
    """
    Возвращает список рекомендованных постов для пользователя на заданную дату и время.
    """
    exp_group = get_exp_group(user_id)
    logger.info(f"применена модель из {exp_group}")
    if exp_group == 'control':
        user_features = user_features_control
        post_features = post_features_control
        feature_order = [
                        'gender', 'age', 'exp_group', 'total_user_viewed', 'total_user_liked',
                        'business_total_post_liked', 'covid_total_post_liked',
                        'entertainment_total_post_liked', 'movie_total_post_liked',
                        'politics_total_post_liked', 'sport_total_post_liked',
                        'tech_total_post_liked', 'age_group', 'topic',
                        'total_post_viewed', 'total_post_liked', 'text_length', 'word_count',
                        'unique_word_count', 'mean_tfidf', 'is_weekend', 'hour', 'period_of_day']
        model = model_control
    elif exp_group == 'test':
        user_features = user_features_test
        post_features = post_features_test
        feature_order = ['gender', 'age', 'exp_group', 'total_user_viewed', 'total_user_liked',
                       'business_total_post_liked', 'covid_total_post_liked',
                       'entertainment_total_post_liked', 'movie_total_post_liked',
                       'politics_total_post_liked', 'sport_total_post_liked',
                       'tech_total_post_liked', 'age_group', 'topic', 'text_emb_0',
                       'text_emb_1', 'text_emb_2', 'text_emb_3', 'text_emb_4', 'text_emb_5',
                       'text_emb_6', 'text_emb_7', 'text_emb_8', 'text_emb_9', 'text_emb_10',
                       'text_emb_11', 'text_emb_12', 'text_emb_13', 'text_emb_14',
                       'total_post_viewed', 'total_post_liked', 'text_length', 'word_count',
                       'unique_word_count', 'is_weekend', 'hour', 'period_of_day']
        model = model_test
    else:
        raise ValueError('unknown group')
    # В этом эндпойнте мы используем ранее загруженную модель и признаки.

    # TODO: извлеките временные признаки из dt, если вы использовали их на этапе обучения модели
    is_weekend = int(dt.weekday() >= 5)
    hour = dt.hour
    if hour < 5:
        period_of_day = 'night'
    elif hour < 12:
        period_of_day = 'morning'
    elif hour < 18:
        period_of_day = 'afternoon'
    else:
        period_of_day = 'evening'

    is_covid_active = 2021 <= dt.year <= 2022
    
    user_row = user_features[user_features['user_id'] == user_id]

    # Если пользователя нет в базе (Fallback) — отдаем глобально популярные посты
    if user_row.empty:
        logger.warning(f"Пользователь {user_id} не найден в БД. Отдаем дефолтные посты.")
        if is_covid_active:
            fallback_posts = post_features.copy()
        else:
            fallback_posts = post_features[post_features['topic'] != 'covid']
            
        fallback_posts = fallback_posts.sort_values(by=['total_post_liked'], ascending=False).head(limit)
        return fallback_posts.rename(columns={"post_id": "id"})[['id', 'text', 'topic']].to_dict(orient="records")

    # === Динамическое формирование списка топиков ===
    topic_like_cols = {
        'business': 'business_total_post_liked',
        'entertainment': 'entertainment_total_post_liked',
        'movie': 'movie_total_post_liked',
        'politics': 'politics_total_post_liked',
        'sport': 'sport_total_post_liked',
        'tech': 'tech_total_post_liked'
    }
    
    if is_covid_active:
        topic_like_cols['covid'] = 'covid_total_post_liked'
    
    # Сортируем активные топики по убыванию лайков пользователя
    user_topic_scores = {topic: user_row[col].values[0] for topic, col in topic_like_cols.items()}
    sorted_topics = sorted(user_topic_scores, key=user_topic_scores.get, reverse=True)

    sampled_posts_list = []
    
    # 1. ПЕРСОНАЛИЗИРОВАННЫЙ ОТБОР (200 постов)
    needed_posts = 200
    for topic in sorted_topics:
        if needed_posts <= 0:
            break
        topic_posts = post_features[post_features['topic'] == topic].sort_values(by='total_post_liked', ascending=False)
        take_count = min(needed_posts, len(topic_posts))
        if take_count > 0:
            sampled_posts_list.append(topic_posts.head(take_count))
            needed_posts -= take_count

    # 2. ДИВЕРСИФИКАЦИЯ: Добор топ-постов из каждой категории (100 постов)
    # Делим 100 постов поровну между всеми доступными топиками
    active_topics = list(topic_like_cols.keys())
    global_posts_per_topic = 100 // len(active_topics)
    
    for topic in active_topics:
        # Берем самые популярные посты топика глобально
        global_topic_posts = (
            post_features[post_features['topic'] == topic]
            .sort_values(by='total_post_liked', ascending=False)
            .head(global_posts_per_topic)
        )
        sampled_posts_list.append(global_topic_posts)

    # Объединяем и удаляем дубликаты (один и тот же пост мог попасть и в персональный топ, и в глобальный)
    if sampled_posts_list:
        candidate_pool = pd.concat(sampled_posts_list, ignore_index=True)
        candidate_pool = candidate_pool.drop_duplicates(subset=['post_id'])
        # Ограничиваем общий пул финально до 300 уникальных постов
        candidate_pool = candidate_pool.sort_values(by='total_post_liked', ascending=False).head(300).reset_index(drop=True)
    else:
        if is_covid_active:
            candidate_pool = post_features.sort_values(by='total_post_liked', ascending=False).head(300)
        else:
            candidate_pool = post_features[post_features['topic'] != 'covid'].sort_values(by='total_post_liked', ascending=False).head(300)

    # === Подготовка признаков для модели ===
    X_predict = candidate_pool.copy()
    
    # Обогащаем признаками пользователя
    for col in user_row.columns:
        if col != 'user_id':
            X_predict[col] = user_row[col].values[0]
            
    # Добавляем временные признаки
    X_predict['is_weekend'] = is_weekend
    X_predict['hour'] = hour
    X_predict['period_of_day'] = period_of_day
    
    # Переводим категориальные колонки в тип 'category' (Критично для HistGBM)
    categorical_cols = ['topic', 'age_group', 'period_of_day']
    for col in categorical_cols:
        if col in X_predict.columns:
            X_predict[col] = X_predict[col].astype('category')
    
    
    X_model_input = X_predict[feature_order]

    # === Предсказание и топ-N ===
    probabilities = model.predict_proba(X_model_input)[:, 1]

    recommendation_pool = (
        X_predict[['post_id', 'text', 'topic']]
        .rename(columns={'post_id': 'id'})
        .copy()
    )
    recommendation_pool['predict_prob'] = probabilities

    top_recommendations = (
        recommendation_pool
        .sort_values('predict_prob', ascending=False)
        .head(limit)
    )

    final_columns = ['id', 'text', 'topic']
    recs = top_recommendations[final_columns].to_dict(orient="records")

    return Response(exp_group=exp_group, recommendations=recs)