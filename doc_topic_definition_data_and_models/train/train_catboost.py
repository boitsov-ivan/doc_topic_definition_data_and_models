import pandas as pd
import numpy as np
import ast
import re
import time


from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report


import pickle
import joblib
from pathlib import Path
from typing import List, Union


import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter




df = pd.read_csv('docs_cleaned.csv')


df['hubs'] = df['hubs'].apply(ast.literal_eval)
df['tags'] = df['tags'].apply(ast.literal_eval)
df['doc_id'] = df['doc_id'].astype(int)


print(f"Размер датафрейма: {df.shape}")
print(f"Колонки: {df.columns.tolist()}")



print('Максимальное число хабов в статье:', df['hubs'].apply(lambda text: len(text)).max())
print('Минимальное число хабов в статье:', df['hubs'].apply(lambda text: len(text)).min())


# бинаризатор для мульти-лейбл классификации
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['hubs'])

# список всех хабов
all_hubs = mlb.classes_
print(f"Всего уникальных хабов: {len(all_hubs)}")
print(f"Примеры хабов: {all_hubs[:30]}")



plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)


print(f"Размер датафрейма: {df.shape}")
print(f"Колонки: {df.columns.tolist()}\n")

# Анализ количества хабов на документ
df['n_hubs'] = df['hubs'].apply(len)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(df['n_hubs'], bins=range(1, df['n_hubs'].max() + 2), edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Количество хабов на документ')
axes[0, 0].set_ylabel('Количество документов')
axes[0, 0].set_title('Распределение количества хабов на документ')
axes[0, 0].axvline(df['n_hubs'].mean(), color='red', linestyle='--', label=f'Среднее: {df["n_hubs"].mean():.2f}')
axes[0, 0].axvline(df['n_hubs'].median(), color='green', linestyle='--', label=f'Медиана: {df["n_hubs"].median():.0f}')
axes[0, 0].legend()


stats_text = f"""
Статистика по хабам на документ:
Максимум: {df['n_hubs'].max()}
Минимум: {df['n_hubs'].min()}
Среднее: {df['n_hubs'].mean():.2f}
Медиана: {df['n_hubs'].median():.0f}
Станд. отклонение: {df['n_hubs'].std():.2f}
"""
axes[0, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[0, 1].axis('off')


# Анализ частоты хабов
all_hubs = [hub for hubs_list in df['hubs'] for hub in hubs_list]
hub_counts = Counter(all_hubs)
hub_freq = pd.DataFrame(hub_counts.most_common(), columns=['hub', 'frequency'])

# Топ-20 хабов
top_n = 20
top_hubs = hub_freq.head(top_n)

axes[1, 0].barh(range(top_n), top_hubs['frequency'].values[::-1])
axes[1, 0].set_yticks(range(top_n))
axes[1, 0].set_yticklabels(top_hubs['hub'].values[::-1])
axes[1, 0].set_xlabel('Частота встречаемости')
axes[1, 0].set_title(f'Топ-{top_n} самых частых хабов')

# Кумулятивное распределение
cumsum = np.cumsum(hub_freq['frequency'].values) / hub_freq['frequency'].sum() * 100
axes[1, 1].plot(range(1, len(cumsum) + 1), cumsum)
axes[1, 1].set_xlabel('Количество уникальных хабов')
axes[1, 1].set_ylabel('Накопленный процент документов (%)')
axes[1, 1].set_title('Кумулятивное распределение хабов')
axes[1, 1].axhline(y=80, color='r', linestyle='--', label='80% документов')
axes[1, 1].axhline(y=95, color='g', linestyle='--', label='95% документов')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


print("СТАТИСТИКА ПО ХАБАМ")
print(f"Всего уникальных хабов: {len(hub_counts)}")
print(f"Всего назначений хабов: {len(all_hubs)}")
print(f"Средняя частота хабов: {np.mean(list(hub_counts.values())):.2f}")
print(f"Медианная частота хабов: {np.median(list(hub_counts.values())):.0f}")

# Анализ разрежённости
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['hubs'])
sparsity = 1 - (y.sum() / (y.shape[0] * y.shape[1]))
print(f"Разрежённость матрицы хабов: {sparsity:.2%}")

# Статистика по разрежённости
docs_per_hub = y.sum(axis=0)
rare_threshold = 10
frequent_threshold = 100

print(f"Классы с частотой < {rare_threshold}: {np.sum(docs_per_hub < rare_threshold)} ({np.sum(docs_per_hub < rare_threshold)/len(docs_per_hub):.1%})")
print(f"Классы с частотой > {frequent_threshold}: {np.sum(docs_per_hub > frequent_threshold)} ({np.sum(docs_per_hub > frequent_threshold)/len(docs_per_hub):.1%})")

# Рекомендации по фильтрации
for threshold in [5, 10, 20, 50, 100, 150, 200, 250, 300]:
    n_classes = np.sum(docs_per_hub >= threshold)
    print(f"При N={threshold:3d} останется классов: {n_classes:4d} ({n_classes/len(docs_per_hub):.1%})")




# Берем только классы, которые встречаются хотя бы N раз
N = 200
frequent_classes = np.where(y.sum(axis=0) >= N)[0]

y = y[:, frequent_classes]
classes_frequent = mlb.classes_[frequent_classes]





X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['text'], y, test_size=0.2, random_state=42, stratify=y.sum(axis=1)
)


vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)



model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='MultiLogloss',
    verbose=50,
    random_seed=42,
    task_type='GPU'
)


print("Распределение классов в y_train:")
for class_idx in range(y_train.shape[1]):  # для каждого класса
    positive_count = y_train[:, class_idx].sum()
    print(f"Класс {class_idx}: {positive_count} положительных примеров из {len(y_train)}")


model.fit(X_train, y_train)



probabilities = model.predict_proba(X_test)
best_threshold = 0.3
y_pred = (probabilities >= best_threshold).astype(int)


print(f"F1-score (micro): {f1_score(y_test, y_pred, average='micro'):.3f}")
print(f"F1-score (macro): {f1_score(y_test, y_pred, average='macro'):.3f}")



mlb_filtered = MultiLabelBinarizer()
mlb_filtered.classes_ = mlb.classes_[frequent_classes]



def get_hub_names(binary_vector):
    return mlb_filtered.inverse_transform(binary_vector.reshape(1, -1))[0]

for i in range(5):
    true_hubs = get_hub_names(y_test[i])
    pred_hubs = get_hub_names(y_pred[i])

    print(f"\nПример {i+1}:")
    print(f"Текст: {X_test_text.iloc[i][:100]}...")
    print(f"Истинные хабы: {list(true_hubs)}")
    print(f"Предсказанные: {list(pred_hubs)}")

    top_indices = probabilities[i].argsort()[-3:][::-1]
    print("Топ-3 вероятности:")
    for idx in top_indices:
        print(f"  {mlb_filtered.classes_[idx]}: {probabilities[i, idx]:.3f}")


model_dir = Path("model_hubs")
model_dir.mkdir(exist_ok=True)


model.save_model(model_dir / "catboost_model.cbm")

joblib.dump(vectorizer, model_dir / "vectorizer.pkl")


with open(model_dir / "mlb_filtered.pkl", 'wb') as f:
    pickle.dump(mlb_filtered, f)


threshold_info = {
    'best_threshold': best_threshold,
    'frequent_classes_indices': frequent_classes,
    'all_classes': mlb.classes_.tolist()
}
with open(model_dir / "threshold_info.pkl", 'wb') as f:
    pickle.dump(threshold_info, f)


print(f"Модель и компоненты сохранены в директорию: {model_dir}")
print(f"Файлы: {list(model_dir.glob('*'))}")





# КЛАСС ДЛЯ ИНФЕРЕНСА

class MultiLabelHubClassifier:
    """
    Класс для инференса модели мульти-лейбл классификации хабов
    """

    def __init__(self, model_dir: Union[str, Path]):
        """
        Загрузка модели и компонентов

        Args:
            model_dir: путь к директории с сохранённой моделью
        """
        self.model_dir = Path(model_dir)

        self.model = CatBoostClassifier()
        self.model.load_model(self.model_dir / "catboost_model.cbm")

        self.vectorizer = joblib.load(self.model_dir / "vectorizer.pkl")

        with open(self.model_dir / "mlb_filtered.pkl", 'rb') as f:
            self.mlb = pickle.load(f)

        with open(self.model_dir / "threshold_info.pkl", 'rb') as f:
            self.threshold_info = pickle.load(f)

        self.threshold = self.threshold_info['best_threshold']

        print(f"Модель успешно загружена из {model_dir}")
        print(f"Количество классов: {len(self.mlb.classes_)}")
        print(f"Используемый порог: {self.threshold}")

    def predict_proba(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Предсказание вероятностей для текстов

        Args:
            texts: один текст или список текстов

        Returns:
            массив вероятностей формы (n_samples, n_classes)
        """
        if isinstance(texts, str):
            texts = [texts]

        X = self.vectorizer.transform(texts)

        probabilities = self.model.predict_proba(X)

        return probabilities

    def predict(self, texts: Union[str, List[str]], threshold: float = None) -> np.ndarray:
        """
        Предсказание бинарных меток для текстов

        Args:
            texts: один текст или список текстов
            threshold: порог для бинаризации (если None, используется сохраненный)

        Returns:
            массив бинарных меток формы (n_samples, n_classes)
        """
        if threshold is None:
            threshold = self.threshold

        probabilities = self.predict_proba(texts)
        predictions = (probabilities >= threshold).astype(int)

        return predictions

    def predict_hub_names(self, texts: Union[str, List[str]], threshold: float = None) -> List[List[str]]:
        """
        Предсказание названий хабов для текстов

        Args:
            texts: один текст или список текстов
            threshold: порог для бинаризации

        Returns:
            список списков с названиями хабов
        """
        if isinstance(texts, str):
            texts = [texts]

        predictions = self.predict(texts, threshold)

        hub_names = []
        for pred in predictions:
            names = self.mlb.inverse_transform(pred.reshape(1, -1))[0]
            hub_names.append(list(names))

        return hub_names

    def predict_with_confidence(self, text: str, top_k: int = 5) -> dict:
        """
        Детальное предсказание для одного текста с топ-k вероятностями

        Args:
            text: входной текст
            top_k: количество топ-классов для вывода

        Returns:
            словарь с предсказаниями и вероятностями
        """
        proba = self.predict_proba(text)[0]
        predictions = (proba >= self.threshold).astype(int)


        pred_hubs = self.mlb.inverse_transform(predictions.reshape(1, -1))[0]


        top_indices = proba.argsort()[-top_k:][::-1]
        top_predictions = [
            {
                'hub': self.mlb.classes_[idx],
                'probability': float(proba[idx])
            }
            for idx in top_indices
        ]

        return {
            'text': text,
            'predicted_hubs': list(pred_hubs),
            'top_predictions': top_predictions,
            'all_probabilities': dict(zip(self.mlb.classes_, proba))
        }

    def evaluate_on_test(self, X_test_text: pd.Series, y_test: np.ndarray) -> dict:
        """
        Оценка модели на тестовых данных

        Args:
            X_test_text: тексты тестовой выборки
            y_test: истинные метки

        Returns:
            словарь с метриками
        """
        y_pred = self.predict(X_test_text)

        metrics = {
            'f1_micro': f1_score(y_test, y_pred, average='micro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'f1_samples': f1_score(y_test, y_pred, average='samples')
        }

        return metrics
    


loaded_classifier = MultiLabelHubClassifier(model_dir)



# обучение по тегам

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['tags'])

# список всех тегов
all_tags = mlb.classes_
print(f"Всего уникальных тегов: {len(all_tags)}")
print(f"Примеры тегов: {all_tags[:50]}")



mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['tags'])

print(f"Всего классов изначально: {y.shape[1]}")


class_counts = y.sum(axis=0)
print(f"Максимальное количество примеров в классе: {class_counts.max()}")
print(f"Минимальное количество примеров в классе: {class_counts.min()}")
print(f"Среднее количество примеров в классе: {class_counts.mean():.1f}")


N = 200
classes_with_enough = (class_counts >= N).sum()
print(f"\nКлассов с ≥ {N} примерами: {classes_with_enough}")

if classes_with_enough == 0:
    print(f"Нет ни одного класса с {N}+ примерами!")
    print(f"Самый частый класс имеет {class_counts.max()} примеров")
    print(f"Уменьшим N до {class_counts.max()} или меньше")



mlb_tags = MultiLabelBinarizer()
y_tags = mlb_tags.fit_transform(df['tags'])


all_tags = mlb_tags.classes_
print(f"Всего уникальных тегов: {len(all_tags)}")
print(f"Примеры тегов: {list(all_tags[:20])}")



tags_counts = y_tags.sum(axis=0)
print(f"Статистика по тегам:")
print(f"  Среднее количество документов на тег: {tags_counts.mean():.1f}")
print(f"  Медиана: {np.median(tags_counts):.1f}")
print(f"  Минимум: {tags_counts.min()}")
print(f"  Максимум: {tags_counts.max()}")


N_tags = 200
frequent_tags_indices = np.where(y_tags.sum(axis=0) >= N_tags)[0]



y_tags_filtered = y_tags[:, frequent_tags_indices]
frequent_tags = mlb_tags.classes_[frequent_tags_indices]

print(f"\nПосле фильтрации (мин. {N_tags} документов):")
print(f"  Осталось тегов: {len(frequent_tags)}")
if len(frequent_tags) > 0:
    print(f"  Примеры: {list(frequent_tags[:20])}")
else:
    print("Нет тегов! Уменьшите порог N_tags")



if len(frequent_tags) == 0:
    raise ValueError("Нет тегов, соответствующих порогу! Уменьшите N_tags.")

# Удаляем документы без тегов после фильтрации
rows_with_tags = y_tags_filtered.sum(axis=1) > 0
print(f"\nДокументов до удаления без тегов: {len(df)}")
print(f"Документов после удаления: {rows_with_tags.sum()}")

df_filtered = df[rows_with_tags].copy()
y_tags_filtered = y_tags_filtered[rows_with_tags]





try:
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df_filtered['text'], y_tags_filtered, test_size=0.2, random_state=42,
        stratify=y_tags_filtered.sum(axis=1)  # Стратификация по количеству тегов
    )
    print(f"\nСтратификация успешно применена")
except ValueError as e:
    print(f"\nСтратификация не удалась: {e}")




print(f"\nРазмер выборки:")
print(f"  Обучающая: {len(X_train_text)}")
print(f"  Тестовая: {len(X_test_text)}")
print(f"  Среднее тегов на документ в обучающей: {y_train.sum(axis=1).mean():.2f}")
print(f"  Среднее тегов на документ в тестовой: {y_test.sum(axis=1).mean():.2f}")




vectorizer_tags = CountVectorizer(
    max_features=5000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95
)

X_train = vectorizer_tags.fit_transform(X_train_text)
X_test = vectorizer_tags.transform(X_test_text)

print(f"\nВекторизация завершена:")
print(f"  Размер словаря: {len(vectorizer_tags.get_feature_names_out())}")
print(f"  Размер матрицы X_train: {X_train.shape}")
print(f"  Размер матрицы X_test: {X_test.shape}")




model_tags = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='MultiLogloss',
    verbose=100,
    random_seed=42,
    early_stopping_rounds=50,
    task_type='GPU'
)


model_tags.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=100)



probabilities_tags = model_tags.predict_proba(X_test)


thresholds = np.arange(0.1, 0.6, 0.05)
best_f1 = 0
best_threshold_tags = 0.3

print("\nПоиск оптимального порога:")
for threshold in thresholds:
    y_pred_temp = (probabilities_tags >= threshold).astype(int)
    f1_micro = f1_score(y_test, y_pred_temp, average='micro', zero_division=0)
    print(f"  Порог {threshold:.2f}: F1-micro = {f1_micro:.3f}")
    if f1_micro > best_f1:
        best_f1 = f1_micro
        best_threshold_tags = threshold

print(f"\nЛучший порог: {best_threshold_tags:.2f} (F1-micro: {best_f1:.3f})")



y_pred_tags = (probabilities_tags >= best_threshold_tags).astype(int)


print(f"\nМетрики на тестовой выборке:")
print(f"  F1-score (micro): {f1_score(y_test, y_pred_tags, average='micro', zero_division=0):.3f}")
print(f"  F1-score (macro): {f1_score(y_test, y_pred_tags, average='macro', zero_division=0):.3f}")
print(f"  F1-score (weighted): {f1_score(y_test, y_pred_tags, average='weighted', zero_division=0):.3f}")
print(f"  F1-score (samples): {f1_score(y_test, y_pred_tags, average='samples', zero_division=0):.3f}")


mlb_tags_filtered = MultiLabelBinarizer()
mlb_tags_filtered.classes_ = frequent_tags

def get_tag_names(binary_vector):
    """Вспомогательная функция для получения имен тегов"""
    return mlb_tags_filtered.inverse_transform(binary_vector.reshape(1, -1))[0]


for i in range(min(5, len(X_test_text))):
    true_tags = get_tag_names(y_test[i])
    pred_tags = get_tag_names(y_pred_tags[i])

    print(f"\nПример {i+1}:")
    print(f"Текст: {X_test_text.iloc[i][:150]}...")
    print(f"Истинные теги: {list(true_tags)}")
    print(f"Предсказанные: {list(pred_tags)}")

    top_indices = probabilities_tags[i].argsort()[-5:][::-1]
    print("Топ-5 тегов по вероятности:")
    for idx in top_indices[:5]:
        if probabilities_tags[i, idx] > 0.1:  # Показываем только значимые
            print(f"  {mlb_tags_filtered.classes_[idx]}: {probabilities_tags[i, idx]:.3f}")


model_dir_tags = Path("model_tags")
model_dir_tags.mkdir(exist_ok=True)

model_tags.save_model(model_dir_tags / "catboost_tags_model.cbm")

joblib.dump(vectorizer_tags, model_dir_tags / "vectorizer_tags.pkl")

with open(model_dir_tags / "mlb_tags_filtered.pkl", 'wb') as f:
    pickle.dump(mlb_tags_filtered, f)

threshold_info_tags = {
    'best_threshold': best_threshold_tags,
    'frequent_tags_indices': frequent_tags_indices,
    'all_tags': mlb_tags.classes_.tolist(),
    'best_f1_score': best_f1,
    'n_tags_filtered': len(frequent_tags),
    'min_frequency': N_tags
}

with open(model_dir_tags / "threshold_tags_info.pkl", 'wb') as f:
    pickle.dump(threshold_info_tags, f)

print(f"\nМодель и компоненты сохранены в директорию: {model_dir_tags}")
print(f"Файлы: {list(model_dir_tags.glob('*'))}")



class MultiLabelTagClassifier:
    """
    Класс модели классификации тегов
    """

    def __init__(self, model_dir: Union[str, Path]):
        """
        Загрузка модели и компонентов из директории

        Args:
            model_dir: путь к директории с сохраненной моделью
        """
        self.model_dir = Path(model_dir)

        self.model = CatBoostClassifier()
        self.model.load_model(self.model_dir / "catboost_tags_model.cbm")

        self.vectorizer = joblib.load(self.model_dir / "vectorizer_tags.pkl")

        with open(self.model_dir / "mlb_tags_filtered.pkl", 'rb') as f:
            self.mlb = pickle.load(f)

        with open(self.model_dir / "threshold_tags_info.pkl", 'rb') as f:
            self.threshold_info = pickle.load(f)

        self.threshold = self.threshold_info['best_threshold']

        print(f"Модель тегов успешно загружена из {model_dir}")
        print(f"Количество тегов: {len(self.mlb.classes_)}")
        print(f"Используемый порог: {self.threshold}")
        print(f"Лучший F1-score при обучении: {self.threshold_info['best_f1_score']:.3f}")

    def predict_proba(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Предсказание вероятностей для текстов

        Args:
            texts: один текст или список текстов

        Returns:
            массив вероятностей формы (n_samples, n_classes)
        """
        if isinstance(texts, str):
            texts = [texts]

        X = self.vectorizer.transform(texts)

        probabilities = self.model.predict_proba(X)

        return probabilities

    def predict(self, texts: Union[str, List[str]], threshold: float = None) -> np.ndarray:
        """
        Предсказание бинарных меток для текстов

        Args:
            texts: один текст или список текстов
            threshold: порог для бинаризации (если None, используется сохраненный)

        Returns:
            массив бинарных меток формы (n_samples, n_classes)
        """
        if threshold is None:
            threshold = self.threshold

        probabilities = self.predict_proba(texts)
        predictions = (probabilities >= threshold).astype(int)

        return predictions

    def predict_tag_names(self, texts: Union[str, List[str]], threshold: float = None) -> List[List[str]]:
        """
        Предсказание названий тегов для текстов

        Args:
            texts: один текст или список текстов
            threshold: порог для бинаризации

        Returns:
            список списков с названиями тегов
        """
        if isinstance(texts, str):
            texts = [texts]

        predictions = self.predict(texts, threshold)

        tag_names = []
        for pred in predictions:
            names = self.mlb.inverse_transform(pred.reshape(1, -1))[0]
            tag_names.append(list(names))

        return tag_names

    def predict_with_confidence(self, text: str, top_k: int = 10) -> dict:
        """
        Детальное предсказание для одного текста с топ-k вероятностями

        Args:
            text: входной текст
            top_k: количество топ-тегов для вывода

        Returns:
            словарь с предсказаниями и вероятностями
        """
        proba = self.predict_proba(text)[0]
        predictions = (proba >= self.threshold).astype(int)

        pred_tags = self.mlb.inverse_transform(predictions.reshape(1, -1))[0]

        top_indices = proba.argsort()[-top_k:][::-1]
        top_predictions = [
            {
                'tag': self.mlb.classes_[idx],
                'probability': float(proba[idx])
            }
            for idx in top_indices
        ]

        return {
            'text': text[:200] + "..." if len(text) > 200 else text,
            'predicted_tags': list(pred_tags),
            'top_predictions': top_predictions,
            'num_tags_predicted': len(pred_tags)
        }

    def evaluate_on_test(self, X_test_text: pd.Series, y_test: np.ndarray) -> dict:
        """
        Оценка модели на тестовых данных

        Args:
            X_test_text: тексты тестовой выборки
            y_test: истинные метки

        Returns:
            словарь с метриками
        """
        y_pred = self.predict(X_test_text)

        metrics = {
            'f1_micro': f1_score(y_test, y_pred, average='micro', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_samples': f1_score(y_test, y_pred, average='samples', zero_division=0),
            'exact_match': np.mean(np.all(y_test == y_pred, axis=1))
        }

        return metrics

    def get_tag_statistics(self, texts: Union[str, List[str]]) -> dict:
        """
        Получение статистики по предсказанным тегам

        Args:
            texts: текст или список текстов

        Returns:
            словарь со статистикой
        """
        predictions = self.predict(texts)
        tag_names = self.predict_tag_names(texts)

        stats = {
            'total_documents': len(texts) if isinstance(texts, list) else 1,
            'total_predictions': sum(len(tags) for tags in tag_names),
            'avg_tags_per_doc': np.mean([len(tags) for tags in tag_names]) if tag_names else 0,
            'unique_tags_predicted': len(set([tag for tags in tag_names for tag in tags])) if tag_names else 0,
            'all_predicted_tags': tag_names
        }

        return stats
    


loaded_tag_classifier = MultiLabelTagClassifier(model_dir_tags)




y_pred_original = (probabilities_tags >= best_threshold_tags).astype(int)
y_pred_loaded = loaded_tag_classifier.predict(X_test_text)


are_equal = np.array_equal(y_pred_original, y_pred_loaded)
print(f"\nПредсказания исходной и загруженной модели идентичны: {are_equal}")
if not are_equal:
    diff_count = (y_pred_original != y_pred_loaded).sum()
    print(f"Количество различий: {diff_count} ({diff_count/y_pred_original.size*100:.2f}%)")


proba_loaded = loaded_tag_classifier.predict_proba(X_test_text)
prob_diff = np.abs(probabilities_tags - proba_loaded).max()
print(f"Максимальная разница в вероятностях: {prob_diff:.10f}")


test_metrics = loaded_tag_classifier.evaluate_on_test(X_test_text, y_test)
print(f"\nМетрики загруженной модели на тестовой выборке:")
print(f"  F1-micro: {test_metrics['f1_micro']:.3f}")
print(f"  F1-macro: {test_metrics['f1_macro']:.3f}")
print(f"  F1-weighted: {test_metrics['f1_weighted']:.3f}")
print(f"  F1-samples: {test_metrics['f1_samples']:.3f}")
print(f"  Exact match (полное совпадение): {test_metrics['exact_match']:.3f}")



for i in range(min(5, len(X_test_text))):
    text = X_test_text.iloc[i]
    true_tags = get_tag_names(y_test[i])


    pred_tags = loaded_tag_classifier.predict_tag_names(text)[0]
    detailed_pred = loaded_tag_classifier.predict_with_confidence(text, top_k=5)

    print(f"\nПример {i+1}:")
    print(f"Текст: {text[:120]}...")
    print(f"Истинные теги: {list(true_tags)}")
    print(f"Предсказанные (загруж. модель): {pred_tags}")
    print(f"Количество предсказанных тегов: {len(pred_tags)}")
    print("Топ-5 тегов по вероятности:")
    for pred in detailed_pred['top_predictions'][:5]:
        print(f"  {pred['tag']}: {pred['probability']:.3f}")



new_text = """
Natural language processing and transformer models like BERT and GPT have revolutionized text classification.
These models use attention mechanisms to understand context. Key applications include sentiment analysis,
named entity recognition, and text summarization. Transfer learning allows fine-tuning on specific tasks.
"""

result = loaded_tag_classifier.predict_with_confidence(new_text, top_k=8)
print(f"\nТекст: {result['text']}")
print(f"Предсказанные теги: {result['predicted_tags']}")
print(f"Количество предсказанных тегов: {result['num_tags_predicted']}")
print("\nТоп-8 тегов по вероятности:")
for pred in result['top_predictions']:
    if pred['probability'] > 0.1:
        print(f"  {pred['tag']}: {pred['probability']:.3f}")



batch_stats = loaded_tag_classifier.get_tag_statistics(X_test_text[:50])
print(f"\nСтатистика по 50 документам:")
print(f"  Всего документов: {batch_stats['total_documents']}")
print(f"  Всего предсказанных тегов: {batch_stats['total_predictions']}")
print(f"  Среднее тегов на документ: {batch_stats['avg_tags_per_doc']:.2f}")
print(f"  Уникальных тегов предсказано: {batch_stats['unique_tags_predicted']}")



n_iterations = 100
start_time = time.time()
for _ in range(n_iterations):
    _ = loaded_tag_classifier.predict(new_text)
single_inference_time = (time.time() - start_time) / n_iterations * 1000

start_time = time.time()
_ = loaded_tag_classifier.predict(X_test_text[:100])
batch_inference_time = (time.time() - start_time) / 100 * 1000

print(f"Среднее время инференса на один текст: {single_inference_time:.2f} мс")
print(f"Среднее время инференса на батч (100 текстов): {batch_inference_time:.2f} мс на текст")
if batch_inference_time > 0:
    print(f"Пропускная способность: {1000/batch_inference_time:.1f} текстов/сек")




if len(frequent_tags) > 0:


    y_pred_final = loaded_tag_classifier.predict(X_test_text)
    tag_precision = []
    tag_recall = []

    for i, tag in enumerate(loaded_tag_classifier.mlb.classes_):
        tp = np.sum((y_test[:, i] == 1) & (y_pred_final[:, i] == 1))
        fp = np.sum((y_test[:, i] == 0) & (y_pred_final[:, i] == 1))
        fn = np.sum((y_test[:, i] == 1) & (y_pred_final[:, i] == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        tag_precision.append(precision)
        tag_recall.append(recall)

        if f1 > 0.3:
            print(f"{tag:30s} - P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f}")

    print(f"\nСредняя precision по всем тегам: {np.mean(tag_precision):.3f}")
    print(f"Средняя recall по всем тегам: {np.mean(tag_recall):.3f}")
else:
    print("\nНет тегов для анализа качества")


















