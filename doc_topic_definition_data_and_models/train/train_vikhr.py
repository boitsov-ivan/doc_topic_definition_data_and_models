import pandas as pd
import numpy as np
import ast

# !pip install -q "transformers>=4.44.0" "accelerate>=0.33.0" "bitsandbytes>=0.43.1" "safetensors" "tqdm" "pandas"

df = pd.read_csv('docs_cleaned.csv')




df['hubs'] = df['hubs'].apply(ast.literal_eval)
df['tags'] = df['tags'].apply(ast.literal_eval)
df['doc_id'] = df['doc_id'].astype(int)


print(f"Размер датафрейма: {df.shape}")
print(f"Колонки: {df.columns.tolist()}")


import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import gc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")
if torch.cuda.is_available():
    print(f"Доступно VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")



model_id = "Vikhrmodels/Vikhr-Gemma-2B-instruct"

print("Загрузка модели Vikhr-Gemma-2B-instruct...")


bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,                    # 8-битная загрузка
    bnb_8bit_quant_type="nf8",            # Normal Float 8
    bnb_8bit_use_double_quant=True,       # двойная квантизация
    bnb_8bit_compute_dtype=torch.bfloat16 # вычисления в bfloat16
)


tokenizer = AutoTokenizer.from_pretrained(model_id)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print("Модель загружена!")
print(f"Использовано VRAM после загрузки: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


def create_summary_prompt(text):
    """
    Создает промпт в формате, ожидаемом моделью Vikhr-Gemma-2B-instruct.
    Формат: <bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
    [citation:4][citation:9]
    """
    # модель поддерживает до 8192 токенов
    if len(text) > 5000:
        text = text[:5000]

    instruction = (
        "Твоя задача — написать краткое и информативное саммари следующего текста. "
        "Выдели главные мысли, ключевые факты и основную идею. "
        "Начинай отвечать сразу с саммари, без вступлений, без фраз 'Саммари:', 'Краткое содержание:' "
        "и без пояснений. Просто напиши сжатый пересказ текста.\n\n"
        f"Текст:\n{text}\n\n"
    )

    # Форматирование для Vikhr-Gemma (как для Gemma 2)
    # <bos> - начало последовательности
    # <start_of_turn>user - начало сообщения пользователя
    # <end_of_turn> - конец сообщения
    # <start_of_turn>model - начало ответа модели
    prompt = f"<bos><start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"

    return prompt



def generate_summary(text, max_new_tokens=300):
    """
    Генерирует саммари для переданного текста.

    Args:
        text: исходный текст
        max_new_tokens: максимальная длина генерируемого саммари

    Returns:
        строка с саммари
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    try:
        prompt = create_summary_prompt(text)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        summary = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        summary = summary.strip()


        if len(summary) < 10:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.9,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.05,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
            generated_tokens = outputs[0][input_length:]
            summary = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        return summary if summary else "Не удалось сгенерировать саммари."

    except Exception as e:
        print(f"Ошибка при генерации саммари: {e}")
        return f"Ошибка: {str(e)[:100]}"



def clear_memory():
    """Очищает кэш GPU для предотвращения OOM"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()



def add_summaries_to_dataframe(df, text_column='text', summary_column='summary', batch_size=5):
    """
    Добавляет колонку с саммари в DataFrame.

    Args:
        df: pandas DataFrame
        text_column: имя колонки с текстом
        summary_column: имя новой колонки для саммари
        batch_size: размер батча (для экономии памяти)

    Returns:
        DataFrame с добавленной колонкой саммари
    """
    summaries = []


    texts = df[text_column].tolist()

    for i, text in enumerate(tqdm(texts, desc="Генерация саммари")):
        summary = generate_summary(text)
        summaries.append(summary)

        if (i + 1) % batch_size == 0:
            clear_memory()

    df[summary_column] = summaries

    clear_memory()

    return df




df_corpus = add_summaries_to_dataframe(df[0:300], text_column='text', summary_column='summary')


df = pd.read_csv('docs_cleaned.csv')
df = df[300:350]
df_test = add_summaries_to_dataframe(df, text_column='text', summary_column='summary')
df_test.to_csv("test_docs_and_summaries.csv", index=False)



# !pip install -q torch>=2.0.0 transformers>=4.40.0 accelerate>=0.33.0 bitsandbytes>=0.43.0 peft>=0.10.0 datasets>=2.18.0 pandas numpy matplotlib seaborn tqdm rouge-score sentence-transformers scikit-learn

# print("Все библиотеки установлены")

# !pip install rouge-score -q

df_corpus = pd.read_csv('docs_and_summaries.csv')

df_test = pd.read_csv('test_docs_and_summaries.csv')





import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

import pandas as pd
import torch
import gc
import re
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
import warnings
warnings.filterwarnings('ignore')


#!pip install rouge-score -q

from rouge_score import rouge_scorer



df_corpus = pd.read_csv('docs_and_summaries.csv')
df_test = pd.read_csv('test_docs_and_summaries.csv')

def clean_summary_prefix(summary):
    """Удаляет различные варианты префиксов из саммари"""
    if not isinstance(summary, str):
        return ""
    patterns = [
        r'^Саммари:\s*\n?', r'Саммари:', r'^Саммари:\s*\*\*?',
        r'^Саммар:\s*\n?', r'^Суммари:\s*\n?', r'^\*\*Краткое содержание:\*\*\s*\n?',
        r'^###\s*\n?', r'^##\s*\n?', r'^---+\s*\n?', r'\*\*Саммар:\*\*',
        r'Краткое содержание:', r'\*\*Суммари:\*\*', r'\*\*\*\*\n'
    ]
    for pattern in patterns:
        summary = re.sub(pattern, '', summary, flags=re.IGNORECASE)
    return summary.strip()

df_corpus['summary'] = df_corpus['summary'].apply(clean_summary_prefix)
df_test['summary'] = df_test['summary'].apply(clean_summary_prefix)


print(f"Исходные данные:")
print(f"  Train samples: {len(df_corpus)}")
print(f"  Test samples: {len(df_test)}")


gc.collect()


def clear_memory():
    """Агрессивная очистка памяти GPU"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

def load_model_4bit_optimized(model_id):
    """Загрузка модели в 4-бит с максимальной экономией памяти"""

    clear_memory()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=True,
        low_cpu_mem_usage=True, 
        torch_dtype=torch.bfloat16
    )

    return model, tokenizer

def setup_lora_model_improved(model):
    """Настройка LoRA с минимальным потреблением памяти"""
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=4,  
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],  
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False

    
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    model.print_trainable_parameters()
    return model

def create_summary_prompt(text, max_len=1500):
    """Создание промпта для модели с ограничением длины"""
    if len(text) > max_len:
        text = text[:max_len].rsplit(' ', 1)[0]

    prompt = f"""<bos><start_of_turn>user
Напиши краткое саммари текста:

{text}

Саммари:<end_of_turn>
<start_of_turn>model
"""
    return prompt

def generate_summary(model, tokenizer, text, max_new_tokens=150):
    """Генерация саммари с очисткой памяти"""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    try:
        prompt = create_summary_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        summary = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        summary = summary.split('<end_of_turn>')[0].strip()
        summary = re.sub(r'^\s*Саммари:\s*', '', summary, flags=re.IGNORECASE)

        if not summary or len(summary) < 10:
            summary = "Краткое содержание не удалось сгенерировать."

        del outputs, inputs
        clear_memory()

        return summary

    except Exception as e:
        print(f"Ошибка генерации: {e}")
        clear_memory()
        return ""



def train_lora_improved(model, tokenizer, df_train, output_dir="./lora_vikhr_improved"):
    """Оптимизированное обучение LoRA с экономией памяти"""

    
    if 'text' not in df_train.columns or 'summary' not in df_train.columns:
        print(f"Колонки не найдены! Доступные колонки: {df_train.columns.tolist()}")
        
        text_col = [col for col in df_train.columns if 'text' in col.lower()][0] if any('text' in col.lower() for col in df_train.columns) else None
        summary_col = [col for col in df_train.columns if 'summ' in col.lower()][0] if any('summ' in col.lower() for col in df_train.columns) else None

        if text_col and summary_col:
            print(f"  Используем колонки: text='{text_col}', summary='{summary_col}'")
            df_train = df_train.rename(columns={text_col: 'text', summary_col: 'summary'})
        else:
            raise ValueError("Не найдены колонки с текстом и саммари")

    df_clean = df_train.dropna(subset=['text', 'summary']).copy()
    df_clean = df_clean[df_clean['text'].astype(str).str.len() > 50]
    df_clean = df_clean[df_clean['summary'].astype(str).str.len() > 20]

    if len(df_clean) < 10:
        print("Мало данных, используем все доступные")
        df_clean = df_train.dropna(subset=['text', 'summary']).head(50)

    if len(df_clean) > 200:
        df_clean = df_clean.sample(n=200, random_state=42)

    print(f"Обучаем на {len(df_clean)} примерах")

    dataset = Dataset.from_pandas(df_clean[['text', 'summary']])


    def process_example(example):
        text = example['text']
        if len(text) > 1000:
            text = text[:1000].rsplit(' ', 1)[0]

        full_text = f"<bos><start_of_turn>user\nНапиши краткое саммари текста:\n{text}\n\nСаммари:<end_of_turn>\n<start_of_turn>model\n{example['summary']}<end_of_turn>"

        tokenized = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=384,
            return_tensors=None
        )

        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(
        process_example,
        remove_columns=['text', 'summary'],
        batched=False
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=5,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=1,
        remove_unused_columns=True,
        report_to="none",
        gradient_checkpointing=True,
        optim="adamw_torch",
        dataloader_drop_last=False
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    print("Начинаем LoRA обучение...")
    trainer.train()


    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"LoRA адаптер сохранен в {output_dir}")
    return model



def evaluate_rouge_improved(df, model, tokenizer, max_samples=15):
    """Оценка ROUGE метрик"""
    df_valid = df.dropna(subset=['text', 'summary'])

    if len(df_valid) == 0:
        return 0.0, 0.0, 0.0

    df_sample = df_valid.sample(n=min(max_samples, len(df_valid)), random_state=42)

    rouge_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    print(f"Оценка на {len(df_sample)} примерах...")

    for idx, row in df_sample.iterrows():
        generated = generate_summary(model, tokenizer, row['text'])
        reference = row['summary']

        if generated and reference and generated != "Краткое содержание не удалось сгенерировать.":
            try:
                scores = scorer.score(reference, generated)
                rouge_scores.append(scores['rouge1'].fmeasure)
            except:
                pass

        clear_memory()

    if rouge_scores:
        avg_rouge = np.mean(rouge_scores)
        print(f"Средний ROUGE-1 F1: {avg_rouge:.4f}")
        return avg_rouge, 0.0, 0.0
    else:
        return 0.0, 0.0, 0.0




clear_memory()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}")
if torch.cuda.is_available():
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA версия: {torch.version.cuda}")

model_id = "Vikhrmodels/Vikhr-Gemma-2B-instruct"


print("\nЗагрузка 4-bit модели...")
try:
    model, tokenizer = load_model_4bit_optimized(model_id)
    print(f"Модель загружена")
except Exception as e:
    print(f"Ошибка загрузки: {e}")
    clear_memory()
    print("Пробуем альтернативный метод загрузки...")
    model, tokenizer = load_model_4bit_optimized(model_id)

clear_memory()


print("Настройка LoRA...")
model = setup_lora_model_improved(model)
clear_memory()


print("\nНачало обучения...")
model = train_lora_improved(model, tokenizer, df_corpus, "./lora_vikhr_improved")
clear_memory()


print("\nОценка на обучающей выборке...")
train_rouge1, _, _ = evaluate_rouge_improved(df_corpus, model, tokenizer, max_samples=15)

print("\nОценка на тестовой выборке...")
test_rouge1, _, _ = evaluate_rouge_improved(df_test, model, tokenizer, max_samples=15)


print("\n" + "="*60)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
print("="*60)
print(f"ROUGE-1 F1 (train): {train_rouge1:.4f}")
print(f"ROUGE-1 F1 (test): {test_rouge1:.4f}")


results = pd.DataFrame({
    'Метрика': ['ROUGE-1 (train)', 'ROUGE-1 (test)'],
    'Значение': [train_rouge1, test_rouge1]
})
results.to_csv('lora_results_improved.csv', index=False)
print("\nРезультаты сохранены")

clear_memory()



# КОД ДЛЯ ЗАГРУЗКИ МОДЕЛИ


import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import re

def clear_memory():
    """Агрессивная очистка памяти GPU"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

def load_model_4bit_optimized(model_id):
    """Загрузка модели в 4-бит с максимальной экономией памяти"""


    clear_memory()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16
    )

    return model, tokenizer

def generate_summary(model, tokenizer, text, max_new_tokens=150):
    """Генерация саммари с защитой от повторений"""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "Текст не предоставлен"

    try:
        if len(text) > 1500:
            text = text[:1500].rsplit(' ', 1)[0]

        prompt = f"""<bos><start_of_turn>user
Напиши краткое и информативное саммари текста.

Текст: {text}

Саммари:<end_of_turn>
<start_of_turn>model
"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                repetition_penalty=1.15,     
                no_repeat_ngram_size=3,      
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                early_stopping=True
            )

        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        summary = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        summary = summary.strip()
        summary = re.sub(r'^\s*Саммари[:\s]*', '', summary, flags=re.IGNORECASE)
        summary = re.sub(r'(\w)\1{5,}', r'\1\1\1', summary)

        if len(summary) < 10 or len(set(summary)) < 3:
            return "Не удалось сгенерировать качественное саммари"

        return summary

    except Exception as e:
        print(f"Ошибка генерации: {e}")
        clear_memory()
        return "Ошибка при генерации"

def load_trained_model(lora_path="./lora_vikhr_improved"):
    """Загрузка обученной модели с LoRA адаптером"""

    print("Начинаем загрузку модели...")
    clear_memory()

    model_id = "Vikhrmodels/Vikhr-Gemma-2B-instruct"

    import os
    if not os.path.exists(lora_path):
        print(f"Папка {lora_path} не найдена!")
        print("Загружаем только базовую модель...")
        model, tokenizer = load_model_4bit_optimized(model_id)
        print("Базовая модель загружена")
        return model, tokenizer


    print("Загрузка базовой модели...")
    model, tokenizer = load_model_4bit_optimized(model_id)

    try:
        print(f"🔧 Загрузка LoRA адаптера из {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()
        print("LoRA адаптер успешно загружен!")

    
        print(f"Состояние: {model.active_adapters}")

    except Exception as e:
        print(f"Не удалось загрузить LoRA адаптер: {e}")
        print("Используем базовую модель без адаптера")

    
    model.config.use_cache = False

    print("Модель готова к работе!")
    return model, tokenizer

def quick_summary(text, model, tokenizer):
    """Быстрая генерация саммари"""
    if not text or len(text) < 10:
        return "Текст слишком короткий"

    summary = generate_summary(model, tokenizer, text)
    return summary


model_infer, tokenizer_infer = load_trained_model("./lora_vikhr_improved")

