"""Script maestro: descargar, procesar y entrenar modelo con Yellow + FHVHV 2023-2025."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def run_command(cmd: list[str], description: str) -> bool:
    """Ejecutar comando shell y reportar resultado."""
    print(f'\n{"="*70}')
    print(f'{description}')
    print('='*70)

    try:
        result = subprocess.run(cmd, check=True, cwd='.')
        return True
    except subprocess.CalledProcessError as e:
        print(f'\n[ERROR] {description} falló con código {e.returncode}')
        return False


def main():
    """Orquestar pipeline completo."""
    print('\n' + '='*70)
    print('PIPELINE MAESTRO: DESCARGAR, PROCESAR Y ENTRENAR')
    print('='*70)

    # Paso 1: Descargar datos
    print('\n[PASO 1] Descargando datos TLC (Yellow + FHVHV, 2023-2025)...')

    # Yellow 2023, 2024, 2025
    cmd = [
        sys.executable, '-m', 'src.data.download_tlc',
        '--dataset', 'yellow',
        '--years', '2023', '2024', '2025'
    ]
    if not run_command(cmd, 'Descargando Yellow Taxis (2023-2025)'):
        return

    # FHVHV 2023, 2024, 2025
    cmd = [
        sys.executable, '-m', 'src.data.download_tlc',
        '--dataset', 'fhvhv',
        '--years', '2023', '2024', '2025'
    ]
    if not run_command(cmd, 'Descargando FHVHV (Uber/Lyft, 2023-2025)'):
        return

    # Paso 2: Procesar datos
    print('\n[PASO 2] Procesando datos (limpieza, agregación)...')

    # Este script no existe aún, lo voy a crear
    cmd = [
        sys.executable, '-m', 'src.data.process_combined'
    ]
    if not run_command(cmd, 'Procesando Yellow + FHVHV combinados'):
        return

    # Paso 3: Feature engineering
    print('\n[PASO 3] Feature engineering...')

    cmd = [
        sys.executable, '-m', 'src.features.build_features_combined'
    ]
    if not run_command(cmd, 'Generando features (Yellow + FHVHV)'):
        return

    # Paso 4: Entrenar modelo
    print('\n[PASO 4] Entrenando modelo...')

    cmd = [
        sys.executable, '-m', 'src.models.train_models_combined'
    ]
    if not run_command(cmd, 'Entrenando modelo (Linear + XGBoost)'):
        return

    # Paso 5: Guardar modelo
    print('\n[PASO 5] Guardando modelo para producción...')

    cmd = [
        sys.executable, '-m', 'src.models.save_model_combined'
    ]
    if not run_command(cmd, 'Guardando modelo'):
        return

    # Paso 6: Evaluar modelo
    print('\n[PASO 6] Evaluando modelo...')

    cmd = [
        sys.executable, '-m', 'src.models.evaluate_model'
    ]
    if not run_command(cmd, 'Evaluación completa'):
        return

    cmd = [
        sys.executable, '-m', 'src.models.detailed_tests'
    ]
    if not run_command(cmd, 'Tests detallados'):
        return

    print('\n' + '='*70)
    print('[OK] PIPELINE COMPLETADO CON EXITO')
    print('='*70)
    print('\nResultados guardados en:')
    print('  - data/processed/ (datos procesados)')
    print('  - models/ (modelo entrenado)')
    print('  - reports/ (gráficos y métricas)')
    print('\nProximo paso: revisar reports/model_metrics_global.csv')


if __name__ == '__main__':
    main()
