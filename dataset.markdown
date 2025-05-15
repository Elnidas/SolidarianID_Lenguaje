# Dataset para Sistema de Recomendación de Causas Solidarias

Se describe un dataset pequeño y estructurado, diseñado para probar un sistema de recomendación de causas solidarias. El dataset incluye usuarios, comunidades y causas, organizados en perfiles temáticos para facilitar la verificación de las recomendaciones.

Este dataset, así como las predicciones, han sido generados con chatGPT, de esta forma tenemos un punto de referencia y comparación

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
