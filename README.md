# American Express - Default Prediction

Репозиторий, посвященный разработке интерпретируемой модели предсказания кредитного риска на [данных соревнования Kaggle](https://kaggle.com/competitions/amex-default-prediction).

## 

- **00.prepare_data.ipynb** - Скрипт подготовки данных
- **01.experiments.ipynb** - Эксперименты с деревьями (в том числе отбор признаков)
- **01.experiments_lr.ipynb** - Эксперименты с моделью логистической регрессии (WoE + binning)
- **01.experiments_nn.ipynb** - Эксперименты с моделью TabNet
- **02.fit_model.ipynb** - Обучение итоговой модели и подбор гиперпараметров
- **import_libs.py** - Список импортируемых библиотек
- **requirments.txt** - Список версий используемых библиотек