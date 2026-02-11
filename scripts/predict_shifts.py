# scripts/predict_shifts.py

"""
Predição de ¹³C-α chemical shifts a partir de estrutura PDB.

Uso:
    python scripts/predict_shifts.py --pdb estrutura.pdb
    python scripts/predict_shifts.py --pdb estrutura.pdb --output resultados.csv
    python scripts/predict_shifts.py --pdb estrutura.pdb --chain B

Input:  Arquivo PDB com coordenadas Cα
Output: CSV com shifts preditos por resíduo
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import load
from scipy.spatial import distance_matrix
import json

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from parsing.pdb_parser import PDBParser


def calculate_geometric_features_single_structure(coords):
    """
    Calcula features geométricas para uma única estrutura.
    
    Args:
        coords: np.ndarray (N, 3) - Coordenadas Cα
    
    Returns:
        pd.DataFrame com features geométricas
    """
    n_residues = len(coords)
    
    # Centroide
    centroid = coords.mean(axis=0)
    
    # Matriz de distâncias
    dist_matrix = distance_matrix(coords, coords)
    
    features = []
    
    for i in range(n_residues):
        # Feature 1: Distância ao centroide
        dist_to_center = np.linalg.norm(coords[i] - centroid)
        
        # Feature 2: Distância ao vizinho mais próximo
        distances_to_others = dist_matrix[i].copy()
        distances_to_others[i] = np.inf
        dist_to_nearest = distances_to_others.min()
        
        # Feature 3: Densidade local (raio 8 Å)
        radius = 8.0
        within_radius = (dist_matrix[i] <= radius).sum() - 1
        local_density = within_radius
        
        # Feature 4: Distância ao N-terminal
        dist_to_n_term = i
        
        # Feature 5: Distância ao C-terminal
        dist_to_c_term = n_residues - 1 - i
        
        features.append({
            'dist_to_center': dist_to_center,
            'dist_to_nearest_neighbor': dist_to_nearest,
            'local_density': local_density,
            'dist_to_n_term': dist_to_n_term,
            'dist_to_c_term': dist_to_c_term
        })
    
    return pd.DataFrame(features)


def load_model_and_metadata(models_dir='models'):
    """
    Carrega modelo treinado, scaler e metadata.
    
    Returns:
        (model, scaler, metadata)
    """
    models_path = Path(models_dir)
    
    if not models_path.exists():
        raise FileNotFoundError(
            f"Diretório {models_dir}/ não encontrado. "
            "Execute train_model.py primeiro para treinar o modelo."
        )
    
    model_file = models_path / 'ridge_model.pkl'
    scaler_file = models_path / 'scaler.pkl'
    metadata_file = models_path / 'metadata.json'
    
    if not model_file.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {model_file}\n"
            "Execute train_model.py para treinar o modelo."
        )
    
    if not scaler_file.exists():
        raise FileNotFoundError(
            f"Scaler não encontrado: {scaler_file}\n"
            "Execute train_model.py para salvar o scaler."
        )
    
    # Carregar modelo e scaler
    print(f"Carregando modelo: {model_file}")
    model = load(model_file)
    
    print(f"Carregando scaler: {scaler_file}")
    scaler = load(scaler_file)
    
    # Carregar metadata (opcional)
    metadata = None
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"Modelo treinado em: {metadata.get('trained_date', 'N/A')}")
        print(f"MAE esperado (CV): {metadata.get('mae_cv', 'N/A'):.2f} ppm")
    
    return model, scaler, metadata


def predict_shifts(pdb_path, chain_id='A', models_dir='models', output_path=None):
    """
    Prediz chemical shifts CA para estrutura PDB.
    
    Args:
        pdb_path: Caminho para arquivo PDB
        chain_id: ID da cadeia a processar
        models_dir: Diretório com modelo treinado
        output_path: Caminho para salvar CSV (opcional)
    
    Returns:
        pd.DataFrame com predições
    """
    print(f"\n{'='*70}")
    print(f"PREDIÇÃO DE CHEMICAL SHIFTS ¹³C-α")
    print(f"{'='*70}\n")
    
    # 1. Carregar modelo
    print("1. Carregando modelo treinado...")
    model, scaler, metadata = load_model_and_metadata(models_dir)
    print("   ✓ Modelo carregado\n")
    
    # 2. Parsear PDB
    print(f"2. Parseando estrutura PDB: {pdb_path}")
    pdb_path = Path(pdb_path)
    
    if not pdb_path.exists():
        raise FileNotFoundError(f"Arquivo PDB não encontrado: {pdb_path}")
    
    parser = PDBParser()
    structure = parser.parse_pdb_file(str(pdb_path), chain_id=chain_id)
    
    print(f"   ✓ PDB ID: {structure.pdb_id}")
    print(f"   ✓ Cadeia: {structure.chain_id}")
    print(f"   ✓ Resíduos: {structure.n_residues}")
    print(f"   ✓ Sequência: {structure.sequence_1letter}\n")
    
    # 3. Calcular features geométricas
    print("3. Calculando features geométricas...")
    coords = structure.ca_coords
    features_df = calculate_geometric_features_single_structure(coords)
    print(f"   ✓ {len(features_df)} conjuntos de features calculados\n")
    
    # 4. Verificar ordem das features
    expected_features = metadata.get('feature_names', [
        'dist_to_center',
        'dist_to_nearest_neighbor',
        'local_density',
        'dist_to_n_term',
        'dist_to_c_term'
    ])
    
    features_df = features_df[expected_features]  # Garantir ordem correta
    
    # 5. Normalizar features
    print("4. Normalizando features...")
    X = features_df.values
    X_scaled = scaler.transform(X)
    print("   ✓ Features normalizadas\n")
    
    # 6. Predizer shifts
    print("5. Predizendo chemical shifts...")
    shifts_predicted = model.predict(X_scaled)
    print(f"   ✓ {len(shifts_predicted)} shifts preditos\n")
    
    # 7. Criar DataFrame de resultados
    results = pd.DataFrame({
        'residue_index': [r.index for r in structure.residues],
        'residue_number': [r.pdb_number for r in structure.residues],
        'residue_type': [r.residue_1letter for r in structure.residues],
        'residue_name': [r.residue_type for r in structure.residues],
        'ca_x': coords[:, 0],
        'ca_y': coords[:, 1],
        'ca_z': coords[:, 2],
        'shift_ca_predicted': shifts_predicted
    })
    
    # 8. Estatísticas
    print("6. Estatísticas das predições:")
    print(f"   Range: {shifts_predicted.min():.2f} - {shifts_predicted.max():.2f} ppm")
    print(f"   Média: {shifts_predicted.mean():.2f} ppm")
    print(f"   Desvio: {shifts_predicted.std():.2f} ppm\n")
    
    # 9. Salvar resultados
    if output_path is None:
        output_path = pdb_path.stem + '_shifts_predicted.csv'
    
    output_path = Path(output_path)
    results.to_csv(output_path, index=False, float_format='%.2f')
    
    print(f"{'='*70}")
    print(f"✓ Predições salvas em: {output_path}")
    print(f"{'='*70}\n")
    
    # 10. Preview
    print("Preview (primeiras 10 linhas):")
    display_cols = ['residue_index', 'residue_type', 'shift_ca_predicted']
    print(results[display_cols].head(10).to_string(index=False))
    print()
    
    return results


def main():
    """Função principal - interface CLI."""
    
    parser = argparse.ArgumentParser(
        description='Prediz chemical shifts ¹³C-α a partir de estrutura PDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python scripts/predict_shifts.py --pdb estrutura.pdb
  python scripts/predict_shifts.py --pdb estrutura.pdb --output resultados.csv
  python scripts/predict_shifts.py --pdb estrutura.pdb --chain B
  python scripts/predict_shifts.py --pdb estrutura.pdb --models models/
        """
    )
    
    parser.add_argument(
        '--pdb',
        required=True,
        type=str,
        help='Arquivo PDB de entrada (obrigatório)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Arquivo CSV de saída (padrão: <pdb_name>_shifts_predicted.csv)'
    )
    
    parser.add_argument(
        '--chain',
        type=str,
        default='A',
        help='ID da cadeia a processar (padrão: A)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        default='models',
        help='Diretório com modelo treinado (padrão: models/)'
    )
    
    args = parser.parse_args()
    
    try:
        results = predict_shifts(
            pdb_path=args.pdb,
            chain_id=args.chain,
            models_dir=args.models,
            output_path=args.output
        )
        
        print("Predição concluída com sucesso!")
        
    except FileNotFoundError as e:
        print(f"\nErro: {e}\n", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nErro inesperado: {e}\n", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()