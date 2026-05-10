import requests
import pytest
import sys
sys.path.append(r'..\sentiment_service\app')
from preprocessing import preprocess_for_llm, filter_quality
import time
import os
from concurrent.futures import ThreadPoolExecutor

session = requests.Session()
session.trust_env = False

os.environ['no_proxy'] = '127.0.0.1,localhost'
os.environ['NO_PROXY'] = '127.0.0.1,localhost'

def test_preprocessing_cleaning():
    """Проверка очистки текста от ссылок и HTML"""
    raw_text = "Привет! <br> Зацени: http://example.com"
    processed = preprocess_for_llm(raw_text)
    assert "http" not in processed
    assert "<br>" not in processed
    assert "Привет" in processed

def test_preprocessing_emojis():
    raw_text = "Круто 😊"
    processed = preprocess_for_llm(raw_text)
    # Проверяем, что пропал сам символ эмодзи и появилось любое описание в двоеточиях или просто текст
    assert "😊" not in processed
    assert "smiling" in processed.lower()

@pytest.mark.parametrize("text, expected", [
    ("Очень крутой товар, рекомендую всем!", True), # Хороший отзыв
    ("!!!", False),                                 # Только знаки
    ("Кратко", False),                              # Слишком короткий
    ("   ", False),                                 # Пустой
])
def test_filter_quality(text, expected):
    """Проверка фильтрации невалидных данных"""
    assert filter_quality(text, min_words=3, min_chars=10) == expected


BASE_URL = "http://127.0.0.1:8000" # Замените на ваш порт

def test_health_check():
    try:
        response = session.get(f"{BASE_URL}/health", timeout=5)
        assert response.status_code == 200
    except requests.exceptions.ConnectionError:
        pytest.fail("Сервер не запущен! Сначала запустите API.")

def test_predict_endpoint():
    """Системный тест: от отправки текста до получения класса"""
    payload = {
        "text": "Ужасный товар",
        "family": "base-rubert"
    }
    response = session.post(f"{BASE_URL}/api/v1/predict", json=payload, timeout=10)
    
    # Если получаем 502, значит прокси все еще мешает или сервер упал
    assert response.status_code == 200, f"Ошибка сервера: {response.status_code}"
    
    data = response.json()
    assert "prediction" in data

def test_model_registration_flow():
    """Тест сценария: регистрация новой модели"""
    payload = {
        "family": "test-family",
        "source_type": "hf",
        "source": "cointegrated/rubert-tiny2",
        "active": True
    }
    response = requests.post(f"{BASE_URL}/api/v1/models/register", json=payload)
    assert response.status_code in [200, 201]
    
    # Проверяем, появилась ли модель в списке
    list_res = requests.get(f"{BASE_URL}/api/v1/families")
    families = [f['family'] for f in list_res.json()['families']]
    assert "test-family" in families


def send_request(_):
    return requests.post(f"{BASE_URL}/api/v1/predict", json={
        "text": "Тестовый запрос для нагрузки",
        "family": "base-rubert"
    }).status_code

def test_load_capacity():
    """Проверка обработки 20 одновременных запросов"""
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(send_request, range(20)))
    
    assert all(code == 200 for code in results)