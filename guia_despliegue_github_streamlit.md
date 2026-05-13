# Guía completa: Subir NÍTIDO a GitHub y desplegarlo en Streamlit Cloud

> Tiempo estimado: **20–30 minutos** la primera vez.

---

## ¿Qué archivos necesitas tener listos?

Antes de empezar, asegúrate de tener estos archivos en una **carpeta local** (p. ej. `nitido_app/`):

```
nitido_app/
├── app.py                    ← La aplicación Streamlit
├── train_model.py            ← Script que entrena y guarda el modelo
├── requirements.txt          ← Dependencias de Python
├── candidatos_nitido.csv     ← Dataset original
├── nitido_reto_ML.ipynb      ← Cuadernillo técnico
├── model_nitido.pkl          ← ⚠️ Se genera corriendo train_model.py
├── scaler_nitido.pkl         ← ⚠️ Se genera corriendo train_model.py
├── explainer_nitido.pkl      ← ⚠️ Se genera corriendo train_model.py
└── feature_cols.pkl          ← ⚠️ Se genera corriendo train_model.py
```

### Paso 0 — Generar los artefactos del modelo (PRIMERO)

Abre una terminal en tu carpeta del proyecto y ejecuta:

```bash
# 1. Instala las dependencias
pip install -r requirements.txt

# 2. Entrena el modelo y genera los .pkl
python train_model.py
```

Verás en consola algo como:
```
Cargando datos...
Train: 3500 | Val: 750 | Test: 750
RF — AUC val: 0.xxxx
Modelo seleccionado: Random Forest (AUC val = 0.xxxx)
✅ Artefactos guardados:
   model_nitido.pkl
   scaler_nitido.pkl
   explainer_nitido.pkl
   feature_cols.pkl
```

---

## PARTE 1: Crear el repositorio en GitHub

### 1.1 Crear cuenta en GitHub (si no tienes)

Ve a [https://github.com](https://github.com) → **Sign up** → sigue los pasos.

---

### 1.2 Crear un repositorio nuevo

1. Una vez en tu perfil de GitHub, haz clic en el botón verde **"New"** (esquina superior izquierda).

2. Configura el repositorio así:

   | Campo | Valor |
   |---|---|
   | Repository name | `nitido-ml` (o el nombre que quieras) |
   | Visibility | **Public** ← Obligatorio para Streamlit Cloud gratis |
   | Initialize with README | ✅ Sí |
   | .gitignore | Python |

3. Haz clic en **"Create repository"**.

---

### 1.3 Instalar Git en tu computador (si no lo tienes)

**Windows:** Descarga desde [https://git-scm.com/download/win](https://git-scm.com/download/win) e instala.  
**Mac:** En la terminal ejecuta `git --version` — si no está, te pedirá instalarlo automáticamente.  
**Linux:** `sudo apt install git`

Configura tu identidad en Git (solo una vez):

```bash
git config --global user.name "Tu Nombre"
git config --global user.email "tu@email.com"
```

---

### 1.4 Clonar el repositorio a tu computador

En la página de tu repositorio en GitHub, haz clic en el botón verde **"Code"** → copia la URL HTTPS.

```bash
# En tu terminal, ve donde quieras guardar el proyecto
cd ~/Desktop

# Clona el repo
git clone https://github.com/TU_USUARIO/nitido-ml.git

# Entra a la carpeta
cd nitido-ml
```

---

### 1.5 Copiar tus archivos al repositorio

Copia todos los archivos de tu carpeta `nitido_app/` dentro de la carpeta `nitido-ml/`:

```bash
# Ejemplo en Mac/Linux
cp ~/ruta/a/nitido_app/app.py .
cp ~/ruta/a/nitido_app/train_model.py .
cp ~/ruta/a/nitido_app/requirements.txt .
cp ~/ruta/a/nitido_app/candidatos_nitido.csv .
cp ~/ruta/a/nitido_app/nitido_reto_ML.ipynb .
cp ~/ruta/a/nitido_app/model_nitido.pkl .
cp ~/ruta/a/nitido_app/scaler_nitido.pkl .
cp ~/ruta/a/nitido_app/explainer_nitido.pkl .
cp ~/ruta/a/nitido_app/feature_cols.pkl .
```

> **En Windows** usa el Explorador de archivos para copiar y pegar directamente en la carpeta `nitido-ml`.

---

### 1.6 Crear el archivo `.gitignore`

Crea un archivo llamado `.gitignore` en la carpeta (o edita el que ya existe) con este contenido:

```
__pycache__/
*.pyc
.env
.DS_Store
*.log
```

> Los archivos `.pkl` **sí** deben subirse al repo para que Streamlit Cloud los encuentre.

---

### 1.7 Subir todo a GitHub

```bash
# Verificar qué archivos están pendientes
git status

# Agregar todos los archivos
git add .

# Hacer el commit
git commit -m "feat: proyecto NÍTIDO - modelo, app y cuadernillo técnico"

# Subir al repositorio remoto
git push origin main
```

Si es la primera vez que haces push, Git puede pedirte autenticación.  
**Opción recomendada:** usa un **Personal Access Token** en lugar de tu contraseña.

Para crearlo: GitHub → tu foto → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token → marca **repo** → Generate. Copia ese token y úsalo como contraseña.

---

### 1.8 Verificar en GitHub

Ve a `https://github.com/TU_USUARIO/nitido-ml` y confirma que ves todos los archivos listados.

---

## PARTE 2: Desplegar en Streamlit Cloud

### 2.1 Crear cuenta en Streamlit Cloud

1. Ve a [https://streamlit.io/cloud](https://streamlit.io/cloud)
2. Haz clic en **"Sign up"**
3. Selecciona **"Continue with GitHub"** → autoriza el acceso
4. Completa tu perfil si te lo pide

---

### 2.2 Crear una nueva app

1. En tu dashboard de Streamlit Cloud, haz clic en **"New app"** (botón azul).

2. Configura así:

   | Campo | Valor |
   |---|---|
   | Repository | `TU_USUARIO/nitido-ml` |
   | Branch | `main` |
   | Main file path | `app.py` |
   | App URL (opcional) | `nitido-candidatos` (o lo que quieras) |

3. Haz clic en **"Deploy!"**

---

### 2.3 Esperar el despliegue

Streamlit Cloud comenzará a:
1. Clonar tu repositorio
2. Instalar las dependencias de `requirements.txt`
3. Ejecutar `app.py`

Este proceso toma entre **2 y 8 minutos** la primera vez.

Verás una pantalla de logs. Si todo sale bien, aparecerá tu app funcionando.

---

### 2.4 Obtener el link público

Una vez desplegada, tu app tendrá una URL pública del tipo:

```
https://TU_USUARIO-nitido-candidatos-app-XXXX.streamlit.app
```

**Copia ese link** — es lo que debes incluir en el cuadernillo técnico y enviar al profesor.

---

## PARTE 3: Actualizar la app (si necesitas hacer cambios)

Si modificas algún archivo (por ejemplo, corriges un bug en `app.py`):

```bash
# 1. Guarda tus cambios
git add .

# 2. Haz el commit con un mensaje descriptivo
git commit -m "fix: corregir cálculo del contrafactual"

# 3. Sube los cambios
git push origin main
```

Streamlit Cloud **detecta automáticamente** el push y re-despliega la app en ~1 minuto.

---

## PARTE 4: Solución de problemas frecuentes

### ❌ "ModuleNotFoundError: No module named 'shap'"

**Causa:** El módulo no está en `requirements.txt`.  
**Solución:** Verifica que `requirements.txt` contenga `shap>=0.45.0` y vuelve a hacer push.

---

### ❌ "FileNotFoundError: model_nitido.pkl not found"

**Causa:** No subiste los archivos `.pkl` al repositorio.  
**Solución:**  
```bash
git add model_nitido.pkl scaler_nitido.pkl explainer_nitido.pkl feature_cols.pkl
git commit -m "add: artefactos del modelo"
git push origin main
```

---

### ❌ La app se cuelga o muestra "Please wait..."

**Causa:** El explainer SHAP con `TreeExplainer` puede tardar varios segundos al arrancar.  
**Solución:** Es normal. Streamlit Cloud carga los artefactos con `@st.cache_resource`, así que solo ocurre una vez por sesión. Si tarda más de 2 minutos, revisa los logs.

---

### ❌ "Authentication failed" al hacer push

**Causa:** GitHub ya no acepta contraseñas por HTTPS.  
**Solución:** Usa un Personal Access Token como contraseña (ver sección 1.7).

---

### ❌ La app funciona local pero no en Streamlit Cloud

**Causa más común:** Diferencia de versiones de librerías.  
**Solución:** Fija las versiones exactas en `requirements.txt`:

```bash
# Obtén las versiones exactas que tienes instaladas
pip freeze | grep -E "streamlit|pandas|numpy|scikit|imbalanced|shap|matplotlib|joblib"
```

Copia el output y reemplaza el contenido de `requirements.txt`.

---

## PARTE 5: Checklist final antes de la sustentación (19 de mayo)

- [ ] Repositorio en GitHub creado y público
- [ ] Todos los archivos subidos (`.py`, `.pkl`, `.csv`, `.ipynb`, `requirements.txt`)
- [ ] App desplegada en Streamlit Cloud y accesible públicamente
- [ ] Link de la app copiado en el cuadernillo técnico
- [ ] La app recibe inputs → muestra predicción + SHAP + contrafactual
- [ ] Probada en un navegador diferente al tuyo (modo incógnito o celular)
- [ ] El cuadernillo técnico (PDF) listo para enviar
- [ ] El informe ejecutivo (PDF) listo para enviar
- [ ] Todo enviado por correo al profesor antes de las 23:59 del 19 de mayo

---

## Estructura final del repositorio en GitHub

```
nitido-ml/
├── app.py                    ← Aplicación Streamlit (Acción 19)
├── train_model.py            ← Script de entrenamiento
├── requirements.txt          ← Dependencias
├── candidatos_nitido.csv     ← Dataset
├── nitido_reto_ML.ipynb      ← Cuadernillo técnico (notebook)
├── model_nitido.pkl          ← Modelo entrenado (serializado)
├── scaler_nitido.pkl         ← Scaler entrenado
├── explainer_nitido.pkl      ← SHAP Explainer
├── feature_cols.pkl          ← Lista de variables
└── README.md                 ← Descripción del proyecto
```

---

*Guía elaborada para el Reto NÍTIDO — Examen 3 — ML II — Universidad Externado de Colombia*
