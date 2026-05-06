# Определение тематики документа с помощью ИИ

## Постановка задачи

Проект по определению тематики документа с помощью ИИ.
Цели проекта:
1) обучить модель для определения тематики документа (мнометочная классификация);
2) обучить модель для составления краткого пересказа содержания документа;
Для обучения моделей используются статьи различных тематик с habr.ru.

## Формат входных и выходных данных

Входные данные:

На вход подаётся статья.

Выходные данные:

Модели определяют тематику документа (метки hubs и tags), генерируют краткий пересказ документа.

## Метрики

В задаче используется метрикики Accuracy, F1-score, ROUGE, BERTScore, V-Measure, Precision@K / Recall@K.

## Валидация

Для тренировки и тестирования данные делятся в соотношении 80/20, для
воспроизводимости зафиксирую конкретный seed 42. Берём от train-а 5% объектов с
помощью функции
[random_split из torch.utils.data.dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split).
Во время валидации мы смотрим, как падает loss и растет accuracy. Для
визуализации используется система логирования [WandB](https://docs.wandb.ai/).

## Данные

### Данные

Датасет - это парсинг статей различных тематик с habr.ru.
Датасет сбалансированный.

### Потенциальные проблемы:

Датасет сбалансированный и достаточно большой. Могут быть проблемы с вычислительными ресурсами для обучения моделей.

## Моделирование

### Бейзлайн

Модели должны справляться лучше чем бейзлайн модели проекта:
1) MultiLabel классификатор по меткам hubs CatBoostClassifier F1-score = 0.61;
2) MultiLabel классификатор по меткам tags CatBoostClassifier F1-score = 0.61;
3) summarization "Vikhr-Gemma-2B-instruct".


### Основные модели

За основу данного проекта взяты CatBoostClassifier и Vikhr-Gemma-2B-instruct.

0. Предобработка данных

Данные из .csv файлов конвертируются в pd.DataFrame, затем посредством класса

```python
class TextDataset(Dataset):
    """Dataset class for torch dataloaders"""

    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return torch.tensor(self.sentences[index], dtype=torch.long), torch.tensor(
            self.labels[index], dtype=torch.long
        )
```

мы преобразуем данные в датасет (только после этого шага можно применить метод
random_split). По всем словам, имеющимся в тренировочном датасете, мы составляем
словарь и токенизируем тексты. Для дальнейшего разбиения на батчи дополняем
короткие строки токеном <pad>. При предобработке тестовых данных неизвестные
слова токенизируются как <unk>.

В данном проекте активно используется библиотека
[PyTorch](https://pytorch.org/), поэтому после предобработки подаём наши данные
в
[torch.nn.DataLoader](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader),
это упрощает написание кода и ускоряет загрузку данных;

Модели:
1) LSTM Classifier для мнометочной классификации по tags и hubs;
2) RNNClassifier для мнометочной классификации по tags и hubs;
1) CatBoostClassifier для мнометочной классификации по tags и hubs;
2) QLoRA-дообучение (4-bit base + LoRA fine-tuning) Vikhr-Gemma-2B-instruct для суммаризации.
    Шаги разработки:
    а) загружаем модель Vikhr-Gemma-2B-instruct в 4-bit → экономия VRAM
    б) добавляем LoRA-адаптеры (ранг 16) в attention и FFN слои
    в) обучаем только адаптеры на маленьком датасете (примеры «текст →
    хорошее саммари»)
    г) сохраняем только LoRA веса (не всю модель)
    д) на инференсе объединяем базовую 4-bit модель + обученный LoRA

В качестве оптимизатора в данной задаче используется
[Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html) с
LR scheduler-ом
[ExponentialLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html);

3. Минимизируем функцию потерь
   [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html);

4. Модели обучаются 10 эпох.

## Внедрение

Модели могут использоваться как пакет для определения тематики документа, пересказа документа. 
В проекте осуществляется перевод обученных моделей в формат .onnx и иные форматы сериализации для дальнейшей работы над моделями как продуктом.

# Работа с проектом

## Setup

Клонирование репозитория:

```
git clone git@github.com:boitsov-ivan/doc_topic_definition_data_and_models.git
```

После клонирования репозитория перейдите в его папку:

```
cd doc_topic_definition_data_and_models
```

Установите Poetry для управления зависимостями:

```
pip install poetry==2.1.2
```

Установка зависимостей проекта:

```
poetry install
```

Установка хуков для pre-commit:

```
poetry run pre-commit install
```

Запустите проверки:

```
cd ..      # перейдите в корень проекта doc_topic_definition_data_and_models
git add .  # добавьте проект в индекс гита
poetry run pre-commit run --all-files
cd doc_topic_definition_data_and_models
```

## Структура папок

```
doc_topic_definition_data_and_models
\----data_load
|        ---download_data.py
|
\----infer
|        ---infer.py
|
\----modules
|        ---compile_to_onnx.py
|
\----train
|        ---train.py
|----config
|
\----data
|        ---\vocab_data
|               ---vocab.json
|
|----models
```
## DVC локальная и на S3

Активируйте окружение из корня проекта:

```
source $(poetry env info --path)/bin/activate
```

Введите ключи доступа к S3 хранилищу в **.dvc/config**

```
[core]
    remote = data

['remote "data"']
    url = ./dvc-storage/data

['remote "models"']
    url = ./dvc-storage/models

['remote "s3-backup"']
    url = s3://mlops-hse-boitsov/dvc-data
    endpointurl = https://storage.yandexcloud.net
    access_key_id =
    secret_access_key =
    region = ru-central1
```


#### Инициализируйте DVC 
```
dvc init
git add .dvc/
git commit -m "Initialize DVC"


mkdir data/s3_data

cp data/*.csv data/s3_data/
cp data/vocab_data data/s3_data/vocab_data

dvc add data/s3_data


mkdir data/s3_models
cp models models/s3_models/

dvc add models/s3_models
```


#### Выгрузить все данные в S3
dvc push -r s3-backup

#### Загрущить данные из S3
dvc push

#### Выгрузить конкретные файлы/папки
dvc push data.dvc -r s3-backup
dvc push models.dvc -r s3-backup

#### Проверить, какие файлы не выгружены
dvc status -c -r s3-backup



## Train

Активируйте окружение из корня проекта:

```
source $(poetry env info --path)/bin/activate
```

Для отслеживания метрик в ходе обучения и валидации необходимо авторизоваться в
WandB. По предложенной ссылке перейдите в личный кабинет, скопируйте и введите
API-ключ в командную строку:

```
poetry run wandb login --relogin
```

Скачайте данные для обучения и тестирования модели:

```
cd doc_topic_definition_data_and_models/data_load

poetry run python download_data.py
```

Запустите модель обучаться:

```
cd ../train

poetry run python train.py     # обучение LSTM Classifier 

Измените настройки в конфиге для обучения RNN Classifier.

Следующий код требует больших вычислительных ресурсов на GPU и модели обучались в гугл колабе:
    1) train_catboost.py    # код обучения CATBOOST классификатора
    2) train_vikhr.py    # код обучения QLora Vikhr-Gemma-2B-instruct
```


## Infer

После того, как модель обучится, запустите систему предсказаний на тестовых
данных docs_test.csv с помощью модели, сохраненной
во время обучения в папке models в файле .ckpt.

Название вашего файла может отличаться. Пример запуска:

```
cd ../infer
poetry run python infer.py docs_test.csv epoch=09-val_loss=0.5523.ckpt
```

Обученную модель можно перевести из .ckpt в формат .onnx. Не забудьте поменять
название файла:

```
cd ../modules

poetry run python compile_to_onnx.py epoch=09-val_loss=0.5523.ckpt
```

## Inference server

Перейдите в корневую директорию проекта и из неё в папку triton.

```
cd triton
```

Скопируйте файл vocab.json из папки data/vocab_data в папку triton.

Установите [Docker](https://www.docker.com/products/docker-desktop/). После
успешной установки убедитесь, что активна сессия:

```
docker ps
```

Соберите образ вашего веб-сервиса:

```
docker-compose build --no-cache web --progress=plain
```

Соберите сервисы:

```
docker-compose build --no-cache
```

Запустите сервисы:

```
docker-compose up
```

Сервер доступен по адресу http://0.0.0.0:8080

Чтобы выйти, нажмите Ctrl+C. Для завершения сессии Docker-а исполните команду

```
docker-compose down
```
