"""
main.py — Pipeline Completo EDA + Evaluación
==============================================
Ejecuta todo el análisis exploratorio y las evaluaciones de modelos.

Uso:
    python src/eda_completo/main.py

Genera:
    src/images/*.png   — Todos los gráficos
    Métricas en consola
"""

import sys
import os

# Asegurar UTF-8 en consola Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Añadir src/eda_completo al path para imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import carga_y_limpieza as paso1
import visualizaciones as paso2
import evaluacion_modelos as paso3
import evaluacion_popularidad as paso4


def main():
    print()
    print("=" * 60)
    print("  PIPELINE COMPLETO — EDA + EVALUACIÓN DE MODELOS")
    print("=" * 60)

    # Paso 1: Carga y limpieza
    df, movies = paso1.run()

    # Paso 2: Visualizaciones EDA
    df = paso2.run(df)

    # Paso 3: Evaluación de 3 modelos (User-CF, Item-CF, SVD)
    paso3.run()

    # Paso 4: Evaluación del sistema de popularidad
    paso4.run()

    print()
    print("=" * 60)
    print("  COMPLETADO — Todos los gráficos generados en src/images/")
    print("=" * 60)


if __name__ == "__main__":
    main()
