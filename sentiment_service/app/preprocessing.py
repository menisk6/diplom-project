from __future__ import annotations

import html
import re
import pandas as pd
import re
import emoji
from sklearn.model_selection import train_test_split

def preprocess_for_llm(text):
    if not isinstance(text, str):
        return ""

    # 1. Обработка эмодзи: конвертируем их в текстовое описание (напр. :smile: -> "smile")
    # Это позволяет модели "читать" эмоцию как слово, что критично для анализа тональности.
    # Используем язык 'ru' для описаний, если это поддерживается, или 'en' (базово).
    text = emoji.demojize(text, delimiters=(" ", " "))

    # 2. Удаление URL и ссылок (не несут смысла для тональности)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 3. Удаление HTML-тегов
    text = re.sub(r'<.*?>', '', text)

    # 4. Нормализация пунктуации
    # Убираем только специфический "мусор", оставляя базовые знаки . , ! ? ; : ( ) -
    # Лишние повторы знаков (!!!!) лучше оставить или сократить до 3-х, так как это маркер экспрессии.
    text = re.sub(r'[^\w\s\d.,!?;:()\-]', ' ', text)

    # 5. Удаление лишних пробелов и переносов строк
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def filter_quality(text, min_words=3, min_chars=10):
    """
    Проверка на качество: удаляем слишком короткие отзывы
    и те, что состоят только из знаков препинания.
    """
    # Если после очистки остались только знаки препинания или пустота
    if not re.search(r'[a-zA-Zа-яА-Я0-9]', text):
        return False
    
    # Проверка на минимальную длину
    words = text.split()
    if len(words) < min_words or len(text) < min_chars:
        return False
    
    return True


def preprocess_text(text: str) -> str:
    if text is None:
        return ''
    text = str(text)
    text = html.unescape(text)
    text = text.replace('\u00a0', ' ')
    text = preprocess_for_llm(text)
    
    if filter_quality(text):
        return text
    else:
        return ''
