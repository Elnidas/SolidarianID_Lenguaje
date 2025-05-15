# Dataset para Sistema de Recomendación de Causas Solidarias

Se describe un dataset pequeño y estructurado diseñado para probar un sistema de recomendación de causas solidarias. El dataset incluye usuarios, comunidades y causas, organizados en perfiles temáticos para facilitar la verificación de las recomendaciones.

Este dataset asi como la predicciones se han realizado con chatGPT, de esta forma tenemos un punto de referencia y comparación

---

## 1. Descripción del Dataset

El dataset contiene:
- **8 usuarios** distribuidos en 4 perfiles temáticos:
  - Educación y Desarrollo Infantil
  - Medioambiente y Sostenibilidad
  - Salud y Bienestar
  - Igualdad y Derechos Sociales
- **6 comunidades**: 4 alineadas con los perfiles de los usuarios y 2 no alineadas o mixtas.
- **12 causas**: 2 por cada comunidad, con descripciones y objetivos relacionados con los temas de las comunidades.

Algunos usuarios tienen intereses **híbridos**, lo que significa que sus bios mencionan más de un tema, permitiendo probar la capacidad del sistema para identificar múltiples afinidades.

---

## 2. Estructura del Dataset

El dataset está dividido en tres archivos JSON:
- **`usuarios.json`**: Contiene la información de los usuarios.
- **`comunidades.json`**: Contiene la información de las comunidades.
- **`causasjson.json`**: Contiene la información de las causas, con `communityId` placeholders que deben actualizarse tras crear las comunidades.

### 2.1. Usuarios (`usuarios.json`)

Cada usuario tiene los siguientes campos:
- `firstName`, `lastName`, `birthDate`, `email`, `password`, `role`, `bio`, `showAge`, `showEmail`.
- El campo `bio` es crucial, ya que describe los intereses del usuario y se usa para las recomendaciones.

**Ejemplo**:
```json
{
  "firstName": "Ana",
  "lastName": "García",
  "birthDate": "1985-05-15",
  "email": "ana.garcia@example.com",
  "password": "Password123",
  "role": "user",
  "bio": "Maestra de primaria apasionada por la educación inclusiva y el desarrollo de habilidades en niños de comunidades marginadas. También me preocupa la salud infantil y el acceso a servicios médicos básicos.",
  "showAge": false,
  "showEmail": false
}
```

### 2.2. Comunidades (`comunidades.json`)

Cada comunidad tiene:
- `name`, `description`, `ownerId`.

**Ejemplo**:
```json
{
  "name": "Educación para Todos",
  "description": "Comunidad dedicada a mejorar el acceso a la educación de calidad en áreas vulnerables, con enfoque en niños y jóvenes.",
  "ownerId": "94175190-0642-492f-afb8-5dc750785f43"
}
```

### 2.3. Causas (`causasjson.json`)

Cada causa tiene:
- `communityId`, `title`, `description`, `startDate`, `endDate`, `objectives`.

**Ejemplo**:
```json
{
  "communityId": "ID_COMUNIDAD_1",
  "title": "Biblioteca Móvil",
  "description": "Proyecto para llevar libros y material educativo a escuelas rurales.",
  "startDate": "2023-01-01T00:00:00.000Z",
  "endDate": "2023-12-31T00:00:00.000Z",
  "objectives": ["QualityEducation"]
}
```

---

## 3. Resultados Esperados

A continuación, se detallan las relaciones esperadas entre usuarios, comunidades y causas, basadas en la similitud entre las bios de los usuarios y las descripciones de las comunidades/causas. Estos resultados nos permitirán verificar si tu sistema de recomendación funciona correctamente.

### 3.1. Recomendación de Causas a Usuarios

Para cada usuario, el sistema debería recomendar las causas más afines según su bio. Aquí algunos ejemplos:

- **Usuario: Ana García** (intereses en educación y salud infantil) 
```plaintext
20ed7f97-4ca6-47f8-a7ca-b9bdebd996c3
```
  - **Causas esperadas**:
    - "Biblioteca Móvil" (Educación)
    - "Becas para Estudiantes" (Educación)
    - "Clínica Móvil" (Salud)
    - "Aulas Digitales" (Educación digital)
  - **Comunidades esperadas**:
    - "Educación para Todos"
    - "Salud Comunitaria"
    - "Tecnología para el Bien"

- **Usuario: Laura Martínez** (intereses en medioambiente y sostenibilidad)
```plaintext
622ac57a-7c7e-43d5-8234-b8363e473865
```
  - **Causas esperadas**:
    - "Paneles Solares Comunitarios"
    - "Talleres de Reciclaje"
  - **Comunidades esperadas**:
    - "Acción Climática Local"

- **Usuario: Diego Fernández** (intereses en energías limpias y educación ambiental) 
```plaintext
920d56cd-15a3-4632-ba55-66c832057101
```
  - **Causas esperadas**:
    - "Paneles Solares Comunitarios"
    - "Talleres de Reciclaje"
    - "Aulas Digitales" (educación digital)
  - **Comunidades esperadas**:
    - "Acción Climática Local"
    - "Tecnología para el Bien"

- **Usuario: Elena Vargas** (intereses en igualdad de género y derechos humanos)
```plaintext
9d40460e-a008-4951-8e77-3e7e0dca6465
```
  - **Causas esperadas**:
    - "Capacitación en Derechos Humanos"
    - "Apoyo a Mujeres Emprendedoras"
  - **Comunidades esperadas**:
    - "Igualdad y Derechos"

### 3.2. Recomendación de Usuarios a Causas

Para una causa dada, el sistema debería sugerir los usuarios más afines. Ejemplos:

- **Causa: "Biblioteca Móvil"** (Educación)
  - **Usuarios esperados**:
    - Ana García
    - Carlos López
    - Diego Fernández (por su interés en educación ambiental)

- **Causa: "Clínica Móvil"** (Salud)
  - **Usuarios esperados**:
    - Ana García (interés en salud infantil)
    - Sofía Ramírez (salud rural)
    - Miguel Torres (salud preventiva)

- **Causa: "Paneles Solares Comunitarios"** (Medioambiente)
  - **Usuarios esperados**:
    - Laura Martínez
    - Diego Fernández

- **Causa: "Capacitación en Derechos Humanos"** (Derechos)
  - **Usuarios esperados**:
    - Elena Vargas
    - Javier Ruiz

---

## Users ID

- 9d40460e-a008-4951-8e77-3e7e0dca6465,Elena,Vargas,1995-02-18,elena.vargas@example.com,Activista por la igualdad de género y los derechos de las minorías. Trabajo en proyectos que fomentan la inclusión social.
- 920d56cd-15a3-4632-ba55-66c832057101,Diego,Fernández,1992-11-25,diego.fernandez@example.com,"Ingeniero especializado en energías limpias. Busco proyectos que impulsen la acción climática y la sostenibilidad, especialmente aquellos que incluyan educación ambiental para jóvenes."
- 843bbb8a-b5fa-42f1-b7c1-fbfb5a1b42f7,Javier,Ruiz,1980-08-22,javier.ruiz@example.com,Abogado especializado en derechos humanos. Apoyo causas que luchan contra la discriminación y promueven la justicia social.,
- 622ac57a-7c7e-43d5-8234-b8363e473865,Laura,Martínez,1988-03-10,laura.martinez@example.com,Ambientalista dedicada a promover energías renovables y prácticas sostenibles en mi comunidad.
- 525a6d3f-1851-4693-970e-b4f47cdbe286,Sofía,Ramírez,1987-09-30,sofia.ramirez@example.com,Enfermera comprometida con mejorar el acceso a la salud en áreas rurales. Participo en campañas de vacunación y educación sanitaria.
- 5199953c-b5e3-4b8b-9b82-00e91d0c2dfc,Carlos,López,1990-07-20,carlos.lopez@example.com,Voluntario en programas de alfabetización para jóvenes. Creo en el poder de la educación para transformar vidas.
- 20ed7f97-4ca6-47f8-a7ca-b9bdebd996c3,Ana,García,1985-05-15,ana.garcia@example.com,Maestra de primaria apasionada por la educación inclusiva y el desarrollo de habilidades en niños de comunidades marginadas. También me preocupa la salud infantil y el acceso a servicios médicos básicos.
- 19fbd526-eee7-4e28-bfc5-905286614f73,Miguel,Torres,1983-12-05,miguel.torres@example.com,Médico voluntario en clínicas móviles. Apoyo iniciativas que promuevan el bienestar integral y la salud preventiva.,

---

## Resultados tras realizar las pruebas

Estas pruebas se han realizado con Ana García cuyo id es: 20ed7f97-4ca6-47f8-a7ca-b9bdebd996c3

**Nota** En estas pruebas Ana **forma parte** de la comunidad **Educación para Todos**, por ese motivo no aparece en las tablas

| Modelo           | Comunidades (Título)                                                                           | Comunidades (Descripción)                                                                         | Comunidades (Título+Descripción)                                                                  | Causas (Título)                                                                                                       | Causas (Descripción)                                                                                            | Causas (Título+Descripción)                                                                                           |
| ---------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **TF-IDF**       | Tecnología para el Bien (0.1722)<br>Salud Comunitaria (0.0812)<br>Arte y Cultura (0.0000)      | Salud Comunitaria (0.2690)<br>Tecnología para el Bien (0.1837)<br>Acción Climática Local (0.1106) | Salud Comunitaria (0.2536)<br>Tecnología para el Bien (0.2040)<br>Acción Climática Local (0.0802) | Talleres de Reciclaje (0.1473)<br>Campaña de Vacunación (0.1473)<br>Festival de Arte Local (0.1243)                   | Talleres de Reciclaje (0.2370)<br>Clínica Móvil (0.2070)<br>Paneles Solares Comunitarios (0.1772)               | Talleres de Reciclaje (0.2117)<br>Clínica Móvil (0.1846)<br>Campaña de Vacunación (0.1333)                            |
| **FastText**     | Tecnología para el Bien (0.8411)<br>Arte y Cultura (0.8182)<br>Igualdad y Derechos (0.8156)    | Salud Comunitaria (0.9721)<br>Tecnología para el Bien (0.9450)<br>Igualdad y Derechos (0.9341)    | Salud Comunitaria (0.9735)<br>Tecnología para el Bien (0.9434)<br>Arte y Cultura (0.9430)         | Campaña de Vacunación (0.8850)<br>Capacitación en Derechos Humanos (0.8713)<br>Apoyo a Mujeres Emprendedoras (0.8661) | Apoyo a Mujeres Emprendedoras (0.9400)<br>Capacitación en Derechos Humanos (0.9315)<br>Aulas Digitales (0.9312) | Capacitación en Derechos Humanos (0.9484)<br>Apoyo a Mujeres Emprendedoras (0.9466)<br>Campaña de Vacunación (0.9414) |
| **beto-uncased** | Tecnología para el Bien (0.6409)<br>Salud Comunitaria (0.5673)<br>Igualdad y Derechos (0.5523) | Salud Comunitaria (0.7654)<br>Igualdad y Derechos (0.7211)<br>Arte y Cultura (0.7192)             | Arte y Cultura (0.7464)<br>Acción Climática Local (0.7347)<br>Salud Comunitaria (0.7144)          | Capacitación en Derechos Humanos (0.6224)<br>Apoyo a Mujeres Emprendedoras (0.5948)<br>Murales Comunitarios (0.5833)  | Capacitación en Derechos Humanos (0.7296)<br>Aulas Digitales (0.7189)<br>Talleres de Reciclaje (0.7101)         | Capacitación en Derechos Humanos (0.7464)<br>Talleres de Reciclaje (0.7371)<br>Aulas Digitales (0.7253)               |
| **beto-cased**   | Tecnología para el Bien (0.8494)<br>Salud Comunitaria (0.8429)<br>Igualdad y Derechos (0.8122) | Salud Comunitaria (0.9201)<br>Acción Climática Local (0.8812)<br>Igualdad y Derechos (0.8776)     | Salud Comunitaria (0.9231)<br>Igualdad y Derechos (0.8950)<br>Tecnología para el Bien (0.8909)    | Capacitación en Derechos Humanos (0.8647)<br>Paneles Solares Comunitarios (0.8155)<br>Becas para Estudiantes (0.8107) | Capacitación en Derechos Humanos (0.9146)<br>Aulas Digitales (0.9139)<br>Campaña de Vacunación (0.8966)         | Capacitación en Derechos Humanos (0.9253)<br>Aulas Digitales (0.9060)<br>Biblioteca Móvil (0.8939)                    |
| **maria**        | Tecnología para el Bien (0.9357)<br>Igualdad y Derechos (0.9330)<br>Arte y Cultura (0.9294)    | Salud Comunitaria (0.9674)<br>Arte y Cultura (0.9587)<br>Tecnología para el Bien (0.9587)         | Salud Comunitaria (0.9632)<br>Tecnología para el Bien (0.9585)<br>Arte y Cultura (0.9583)         | Capacitación en Derechos Humanos (0.9362)<br>Telemedicina Rural (0.9265)<br>Apoyo a Mujeres Emprendedoras (0.9249)    | Aulas Digitales (0.9658)<br>Biblioteca Móvil (0.9654)<br>Capacitación en Derechos Humanos (0.9647)              | Capacitación en Derechos Humanos (0.9675)<br>Biblioteca Móvil (0.9670)<br>Aulas Digitales (0.9617)                    |

### Comunidades

- Todos los modelos identifican “Salud Comunitaria” y “Tecnología para el Bien” como afines (coinciden con los intereses en salud infantil y educación/tecnología de Ana). Aunque **Maria** y **FastText** no lo asocian basándose únicamente en el título, algo similar ocurre con **beto** con **Tecnología para el Bien**, que según el contexto no la recomienda
- Como comunidad extra algunos modelos recomiendan **Igualdad y Derechos** o **Arte y Cultura**

###  Causas

- Éxito esperado: “Biblioteca Móvil”, “Becas para Estudiantes”, “Clínica Móvil”, “Aulas Digitales”.
- Solo beto-cased y maria recuperan dos de ellas en su Top-3 (“Biblioteca Móvil” y “Aulas Digitales”).
- TF-IDF acierta “Clínica Móvil” pero mezcla temas ambientales (“Talleres de Reciclaje”).
- FastText y beto-uncased sesgan hacia “Derechos Humanos / Mujeres” por similitud semántica con “comunidades marginadas” y “desarrollo” de la bio.

### Conclusiones

- Las puntuaciones de maria son, en general, las más altas (≈0.93-0.97), signo de mayor “confianza” interna.
- TF-IDF muestra la caída más pronunciada: su Top-3 baja de ≈0.17 a 0.08 en comunidades, revelando poca separación entre candidatos.
- Los modelos beto-cased y Maria son claramente los más robustos para este caso. Logran alinear adecuadamente los intereses expresados en lenguaje natural con causas específicas, respetando tanto el contenido como la intención del perfil. FastText y beto-uncased tienden a sesgar las recomendaciones hacia temas sociales más amplios por asociación semántica laxa, mientras que TF-IDF carece de capacidad interpretativa profunda. 


---



## beto_test_metrics


| label | precision | recall | f1-score | support |
|-------|-----------|--------|----------|---------|
| 0 | 0.9043 | 0.9512 | 0.9272 | 10949 |
| 1 | 0.8049 | 0.6666 | 0.7292 | 3305 |
| accuracy |  |  | 0.8852 | 14254 |
| macro avg | 0.8546 | 0.8089 | 0.8282 | 14254 |
| weighted avg | 0.8813 | 0.8852 | 0.8813 | 14254 |

| | pred 0 | pred 1 |
|---|-------|-------|
| real 0 | 10415 | 534 |
| real 1 | 1102 | 2203 |

**Tiempo inferencia:** 57.60 s (0.0040 s/ej.)  
**Pico CPU:** 8.3 MB  (RSS final 783.1 MB)  
**Pico GPU:** 1548.4 MB  

---

## distilbeto_test_metrics


| label | precision | recall | f1-score | support |
|-------|-----------|--------|----------|---------|
| 0 | 0.9236 | 0.9318 | 0.9277 | 10949 |
| 1 | 0.7671 | 0.7446 | 0.7557 | 3305 |
| accuracy |  |  | 0.8884 | 14254 |
| macro avg | 0.8454 | 0.8382 | 0.8417 | 14254 |
| weighted avg | 0.8873 | 0.8884 | 0.8878 | 14254 |

| | pred 0 | pred 1 |
|---|-------|-------|
| real 0 | 10202 | 747 |
| real 1 | 844 | 2461 |

**Tiempo inferencia:** 29.80 s (0.0021 s/ej.)  
**Pico CPU:** 8.3 MB  (RSS final 795.9 MB)  
**Pico GPU:** 1385.9 MB  

---

## maria_test_metrics


| label | precision | recall | f1-score | support |
|-------|-----------|--------|----------|---------|
| 0 | 0.9195 | 0.9414 | 0.9303 | 10949 |
| 1 | 0.7892 | 0.7271 | 0.7569 | 3305 |
| accuracy |  |  | 0.8917 | 14254 |
| macro avg | 0.8543 | 0.8342 | 0.8436 | 14254 |
| weighted avg | 0.8893 | 0.8917 | 0.8901 | 14254 |

| | pred 0 | pred 1 |
|---|-------|-------|
| real 0 | 10307 | 642 |
| real 1 | 902 | 2403 |

**Tiempo inferencia:** 58.59 s (0.0041 s/ej.)  
**Pico CPU:** 8.3 MB  (RSS final 801.2 MB)  
**Pico GPU:** 1605.4 MB  

---

## tfidf_linear_svc


| label | precision | recall | f1-score | support |
|-------|-----------|--------|----------|---------|
| 0 | 0.9126 | 0.8930 | 0.9027 | 10949 |
| 1 | 0.6691 | 0.7165 | 0.6920 | 3305 |
| accuracy |  |  | 0.8521 | 14254 |
| macro avg | 0.7908 | 0.8048 | 0.7973 | 14254 |
| weighted avg | 0.8561 | 0.8521 | 0.8538 | 14254 |

| | pred 0 | pred 1 |
|---|-------|-------|
| real 0 | 9778 | 1171 |
| real 1 | 937 | 2368 |

**Tiempo inferencia:** 1.25 s (0.0001 s/ej.)  
**Pico CPU:** 13.1 MB  (RSS final 560.9 MB)  

---


## tfidf_logreg 


| label | precision | recall | f1-score | support |
|-------|-----------|--------|----------|---------|
| 0 | 0.9183 | 0.8741 | 0.8957 | 10949 |
| 1 | 0.6403 | 0.7422 | 0.6875 | 3305 |
| accuracy |  |  | 0.8436 | 14254 |
| macro avg | 0.7793 | 0.8082 | 0.7916 | 14254 |
| weighted avg | 0.8538 | 0.8436 | 0.8474 | 14254 |

| | pred 0 | pred 1 |
|---|-------|-------|
| real 0 | 9571 | 1378 |
| real 1 | 852 | 2453 |

**Tiempo inferencia:** 1.18 s (0.0001 s/ej.)  
**Pico CPU:** 13.1 MB  (RSS final 560.4 MB)  

---

## tfidf_multinb 


| label | precision | recall | f1-score | support |
|-------|-----------|--------|----------|---------|
| 0 | 0.8196 | 0.9894 | 0.8965 | 10949 |
| 1 | 0.8881 | 0.2787 | 0.4242 | 3305 |
| accuracy |  |  | 0.8246 | 14254 |
| macro avg | 0.8539 | 0.6340 | 0.6604 | 14254 |
| weighted avg | 0.8355 | 0.8246 | 0.7870 | 14254 |

| | pred 0 | pred 1 |
|---|-------|-------|
| real 0 | 10833 | 116 |
| real 1 | 2384 | 921 |

**Tiempo inferencia:** 1.18 s (0.0001 s/ej.)  
**Pico CPU:** 13.1 MB  (RSS final 557.7 MB)  

---

## tfidf_rf 


| label | precision | recall | f1-score | support |
|-------|-----------|--------|----------|---------|
| 0 | 0.8484 | 0.9780 | 0.9086 | 10949 |
| 1 | 0.8523 | 0.4209 | 0.5635 | 3305 |
| accuracy |  |  | 0.8488 | 14254 |
| macro avg | 0.8503 | 0.6994 | 0.7360 | 14254 |
| weighted avg | 0.8493 | 0.8488 | 0.8286 | 14254 |

| | pred 0 | pred 1 |
|---|-------|-------|
| real 0 | 10708 | 241 |
| real 1 | 1914 | 1391 |

**Tiempo inferencia:** 1.36 s (0.0001 s/ej.)  
**Pico CPU:** 18.4 MB  (RSS final 1190.1 MB)  
