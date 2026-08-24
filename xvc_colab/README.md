# X-VC Colab + Gradio

Экспериментальный Google Colab для zero-shot voice conversion через официальный X-VC.

## Что есть

- Source Audio и Reference Audio через Gradio
- offline-конвертация по умолчанию
- streaming simulation с параметрами X-VC
- модель загружается один раз и переиспользуется между конвертациями
- готовый установщик для Colab
- автоматическая загрузка официального X-VC checkpoint, GLM-4-Voice tokenizer и ERes2Net speaker encoder

## Запуск в Colab

1. Открой `X_VC_Colab.ipynb`.
2. Выбери T4 GPU.
3. Нажми Выполнить всё.
4. Открой публичную Gradio-ссылку.
5. Загрузи source и reference.
6. Для первого теста оставь Offline quality.

## Upstream

Официальный X-VC фиксируется на commit:

`49df8c591eafc48b096e466d96f9839f9c0dd739`

Это текущий опубликованный commit официального репозитория Jerrister/X-VC на момент сборки.

## Важно про русский

Опубликованные результаты X-VC в основном относятся к английскому и китайскому. Этот Colab нужен именно для честного практического теста русского source и русского reference, без предварительного обучения.
