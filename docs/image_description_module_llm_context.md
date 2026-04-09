# Contexto para pedir a un LLM un modulo de descripcion de imagenes

## Objetivo

Quiero extender este proyecto para agregar un modulo que describa el contenido de cada imagen en lenguaje natural.

El proyecto ya tiene codigo para:

- leer imagenes PNG manualmente
- convertir pixeles RGB a escala de grises
- calcular estadisticas basicas y rasgos forenses
- extraer features numericas para analisis
- hacer una clasificacion simple con KNN

No quiero que el LLM me proponga algo desde cero ignorando lo que ya existe. Quiero que revise mi estructura actual y me diga:

1. que partes de mi codigo actual puedo reutilizar
2. que partes no sirven para descripcion semantica de imagenes
3. que archivos o carpetas deberia agregar
4. como conectar el nuevo modulo con mi pipeline actual
5. una propuesta incremental, simple de implementar

## Estructura actual relevante

```text
backend/ml/
  features/ 
    feature_extractor.py
  forensics/
    basic_stats.py
    block_analysis.py
    knn.py
    raw_image_reader.py
    tests.py
  inference/
  models/
  tests/
    test_feature_extractor.py
  training/
```

## Archivos que ya existen y que hacen

### 1. `backend/ml/forensics/raw_image_reader.py`

Contiene una clase `PNGReader` que:

- valida la firma del archivo PNG
- lee los chunks principales
- parsea `IHDR`
- descomprime `IDAT`
- reconstruye pixeles RGB manualmente

Limitaciones actuales:

- solo soporta PNG RGB (`color_type == 2`)
- no parece estar pensado para JPG, JPEG, WEBP u otros formatos

Esto me sirve como lector base, pero necesito saber si conviene:

- reutilizarlo tal cual para un primer MVP
- envolverlo con una interfaz generica de carga de imagen
- o reemplazarlo por una libreria mas flexible para el modulo de descripcion

### 2. `backend/ml/forensics/basic_stats.py`

Contiene funciones para calculo numerico sobre imagenes:

- `rgb_to_grayscale(pixels, width)`
- `mean(matrix)`
- `variance(matrix, mean_value=None)`
- `entropy(matrix)`
- `laplacian_filter(matrix)`
- `residual_energy(matrix)`
- `dft_2d(matrix)`
- `spectral_energy(spectrum)`
- `gradient_magnitude(matrix)`
- `gradient_stats(matrix)`
- `ai_derivative_kernel_score(matrix)`

Estas funciones estan orientadas a:

- textura
- energia
- bordes
- frecuencia
- analisis forense

No describen semanticamente el contenido de una imagen. Quiero que me indiques si alguna puede reutilizarse como metadato auxiliar para enriquecer una descripcion, por ejemplo:

- "imagen oscura"
- "alto contraste"
- "mucho detalle"
- "escena uniforme"

### 3. `backend/ml/forensics/block_analysis.py`

Contiene:

- `split_into_blocks(matrix, block_size=32)`
- `block_energy(block)`
- `energy_distribution_variance(blocks)`

Esto parece servir para medir variacion local de energia por bloques. Necesito saber si vale la pena integrarlo al futuro modulo de descripcion o si deberia quedarse solo como parte forense.

### 4. `backend/ml/features/feature_extractor.py`

Contiene una clase `FeatureExtractor` con el metodo:

- `extract_from_png_reader(reader)`

Ese metodo:

- convierte a grayscale
- calcula media, varianza y entropia
- aplica laplaciano y energia residual
- analiza bloques
- calcula estadisticas de gradiente
- calcula energia espectral con DFT sobre subimagen 64x64
- calcula una energia normalizada basada en segunda derivada

Regresa un diccionario con features como:

- `mean`
- `variance`
- `entropy`
- `hf_energy`
- `energy_to_variance_ratio`
- `block_energy_variance`
- `gradiente_mean`
- `gradient_variance`
- `spectral_energy`
- `energia_diagonal_2da_derivada_norm`

Este archivo me parece la mejor base para reutilizar el pipeline actual. Quiero saber si deberia:

- extender esta clase
- crear otro extractor paralelo
- o separar rasgos forenses de rasgos para descripcion semantica

### 5. `backend/ml/forensics/knn.py`

Contiene una implementacion manual de `KNN`:

- `fit(X, y)`
- `predict(X)`

La distancia es euclidiana y la salida parece binaria (`0` o `1`).

Esto lo estoy usando para clasificacion simple, no para descripcion de imagen. Quiero que me digas si:

- este archivo no participa en el nuevo modulo
- podria servir para una etapa auxiliar
- o deberia mantenerse separado

### 6. `backend/ml/forensics/tests.py`

Este script:

- carga un dataset
- extrae features simples
- normaliza
- divide train/test
- entrena KNN
- calcula accuracy

Parece mas un experimento de entrenamiento que parte del producto final.

### 7. `backend/ml/tests/test_feature_extractor.py`

Este script:

- recorre un folder
- procesa archivos PNG
- usa `PNGReader`
- llama `FeatureExtractor.extract_from_png_reader`
- imprime features por imagen

Me sirve como ejemplo de flujo actual:

`imagen -> reader -> feature extractor -> resultados`

## Lo que creo que me falta

Creo que para describir contenido de imagenes me faltan varias piezas que no tengo claras:

- un modulo semantico o vision-language
- una interfaz para generar una descripcion tipo caption
- posiblemente soporte para mas formatos de imagen
- un lugar en la arquitectura para inferencia de descripciones
- validacion o pruebas del nuevo flujo

## Lo que quiero que me propongas

Con base en la estructura y archivos que ya tengo, proponme una solucion concreta para agregar descripcion de imagenes.

Quiero que tu respuesta incluya:

1. una evaluacion de que codigo actual reutilizar y que codigo no mezclar
2. una arquitectura minima viable
3. nombres de archivos y carpetas nuevos sugeridos
4. firmas de clases o funciones recomendadas
5. flujo de datos de extremo a extremo
6. si conviene mantener separado:
   - analisis forense
   - extraccion de features
   - descripcion semantica
7. si puedo aprovechar mis estadisticas basicas para enriquecer el texto final
8. una estrategia simple primero y luego una estrategia mejorada

## Restricciones deseadas

- quiero reutilizar lo existente cuando tenga sentido
- prefiero cambios incrementales
- no quiero romper el pipeline actual
- me gustaria una solucion clara para backend en Python
- si propones nuevas carpetas, ubicalas dentro de mi estructura actual

## Pregunta principal para el LLM

Con esta base, como agregarias un modulo de descripcion de imagenes que aproveche mi codigo actual sin confundir el analisis forense con la descripcion semantica, y que archivos nuevos agregarias dentro de esta estructura?
