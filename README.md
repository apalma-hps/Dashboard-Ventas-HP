
# Dashboard Streamlit + Google Sheets (OAuth) — PyCharm

## 1) Abrir en PyCharm
- Open > selecciona esta carpeta.

## 2) Crear venv e instalar dependencias
PyCharm suele sugerir el entorno virtual automáticamente. Si no:
- File > Settings > Project > Python Interpreter > Add… > Virtualenv.
Luego instala:
```
pip install -r requirements.txt
```

## 3) Credenciales OAuth
- Crea credenciales tipo **"Aplicación de escritorio"** en https://console.cloud.google.com/apis/credentials
- Descarga el archivo y renómbralo **`client_secret.json`**.
- Colócalo en la **raíz del proyecto** (junto a `app_oauth.py`).

## 4) Configurar Run/Debug en PyCharm
- Run > Edit Configurations… > (+) **Python**
  - Name: `Streamlit`
  - Script path: ruta hacia el ejecutable de streamlit (ej. en macOS: `.venv/bin/streamlit`)
    - Alternativa: usa `Module name` = `streamlit`
  - Parameters: `run app_oauth.py`
  - Working directory: esta carpeta del proyecto
  - (Opcional) Emulate terminal in output console: ON
- Apply > OK

## 5) Ejecutar
- Run ▶ `Streamlit` (la config que creaste).
- Se abrirá tu navegador para autorizar. Se guardará `token.json`.

## 6) Usar el dashboard
- Pega la URL del Google Sheets y el nombre de la pestaña (worksheet).

## Notas
- Si ves `redirect_uri_mismatch`, asegúrate que las credenciales sean **Aplicación de escritorio** (no web).
- Si el Sheet está en otra cuenta, comparte acceso a tu cuenta (la que usaste en OAuth).
