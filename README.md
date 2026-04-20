# CycleGAN Interiors

Простое Streamlit-демо для перевода интерьеров между доменами `modern` и `rustic`.

Что внутри:

- `cycle_gan_wetrythebest.pt` — checkpoint модели
- `cyclegan_interiors.py` — модель и инференс
- `streamlit_app.py` — интерфейс
- `app.py` — точка входа для Streamlit Cloud

Локальный запуск:

```bash
pip install -r requirements.txt
streamlit run app.py
```
