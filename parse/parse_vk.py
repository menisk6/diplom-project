import vk_api
import json
import re
import time
from datetime import datetime

# Настройки
ACCESS_TOKEN = 'ACCESS_TOKEN'
OUTPUT_PATH = r"..\datasets\vk_comments.json"

URLS = {
    "https://vk.com/topic-202101115_48260487": "Milana Shop. Садовод 18-17",
    "https://vk.com/topic-214178039_48908065": "VES SHOP - оригинальная брендовая одежда",
    "https://vk.com/topic-204631922_48114885": "Style Shop | Садовод корпус Б 2В-23-25",
    "https://vk.com/topic-34070005_27001432": "YOLO SHOP"
}

def parse_topic_url(url):
    """Извлекает ID группы и ID темы из ссылки"""
    match = re.search(r"topic-(\d+)_(\d+)", url)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def main():
    vk_session = vk_api.VkApi(token=ACCESS_TOKEN)
    vk = vk_session.get_api()

    for url, shop_name in URLS.items():
        group_id, topic_id = parse_topic_url(url)
        if not group_id:
            print(f"Ошибка парсинга URL: {url}")
            continue

        print(f"Парсинг: {shop_name} (ID: {topic_id})")
        
        offset = 0
        count = 100
        topic_comments = []
        author_ids = set()

        while True:
            try:
                response = vk.board.getComments(
                    group_id=group_id,
                    topic_id=topic_id,
                    count=count,
                    offset=offset,
                    sort='asc'
                )
                
                items = response.get('items', [])
                if not items:
                    break

                for item in items:
                    # Собираем ID авторов для последующего получения имен
                    author_ids.add(item['from_id'])
                    
                    # Форматируем дату как в Selenium (или ISO)
                    date_str = datetime.fromtimestamp(item['date']).strftime('%d %b %Y в %H:%M')
                    
                    topic_comments.append({
                        "date": date_str,
                        "text": item['text'],
                        "shop": shop_name
                    })

                offset += count
                if offset >= response['count']:
                    break
                
                time.sleep(0.34) # Защита от Flood Control

            except Exception as e:
                print(f"Ошибка при сборе комментариев: {e}")
                break

        print(f"Найдено комментариев: {len(topic_comments)}")

    # Сохраняем в JSON
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(topic_comments, f, ensure_ascii=False, indent=4)
        print(f"\nГотово! Результаты сохранены в {OUTPUT_PATH}")
    except FileNotFoundError:
        print(f"Ошибка: Путь {OUTPUT_PATH} не найден. Проверьте существование папки.")

if __name__ == "__main__":
    main()