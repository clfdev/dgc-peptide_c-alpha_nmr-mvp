# scripts/predict_shifts.py

"""
Predição de ¹³C-α chemical shifts a partir de estrutura PDB.

Versão 2: 27 features expandidas (12 geométricas + 15 físico-químicas)

Uso:
    python scripts/predict_shifts.py --pdb estrutura.pdb
    python scripts/predict_shifts.py --pdb estrutura.pdb --output resultados.csv
    python scripts/predict_shifts.py --pdb estrutura.pdb --chain B

    Exemplo: python ./scripts/predict_shifts.py --pdb 1L2Y.pdb

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


# Propriedades físico-químicas dos aminoácidos (AAindex scales)
AMINO_ACID_PROPERTIES = {
    # [hydrophobicity, volume, charge, polarity, aromaticity, 
    #  flexibility, accessibility, helix_prop, sheet_prop, turn_prop]
    'A': [1.8, 88.6, 0, 0, 0, 0.360, 0.62, 1.42, 0.83, 0.66],
    'C': [2.5, 108.5, 0, 0, 0, 0.346, 0.29, 0.70, 1.19, 1.19],
    'D': [-3.5, 111.1, -1, 1, 0, 0.511, 0.90, 1.01, 0.54, 1.46],
    'E': [-3.5, 138.4, -1, 1, 0, 0.497, 0.75, 1.51, 0.37, 0.74],
    'F': [2.8, 189.9, 0, 0, 1, 0.314, 0.42, 1.13, 1.38, 0.60],
    'G': [-0.4, 60.1, 0, 0, 0, 0.544, 0.81, 0.57, 0.75, 1.56],
    'H': [-3.2, 153.2, 0.5, 1, 1, 0.323, 0.67, 1.00, 0.87, 0.95],
    'I': [4.5, 166.7, 0, 0, 0, 0.462, 0.46, 1.08, 1.60, 0.47],
    'K': [-3.9, 168.6, 1, 1, 0, 0.466, 0.85, 1.16, 0.74, 1.01],
    'L': [3.8, 166.7, 0, 0, 0, 0.365, 0.45, 1.21, 1.30, 0.59],
    'M': [1.9, 162.9, 0, 0, 0, 0.295, 0.40, 1.45, 1.05, 0.60],
    'N': [-3.5, 114.1, 0, 1, 0, 0.463, 0.78, 0.67, 0.89, 1.56],
    'P': [-1.6, 112.7, 0, 0, 0, 0.509, 0.76, 0.57, 0.55, 1.52],
    'Q': [-3.5, 143.8, 0, 1, 0, 0.493, 0.84, 1.11, 1.10, 0.98],
    'R': [-4.5, 173.4, 1, 1, 0, 0.529, 0.95, 0.98, 0.93, 0.95],
    'S': [-0.8, 89.0, 0, 1, 0, 0.507, 0.78, 0.77, 0.75, 1.43],
    'T': [-0.7, 116.1, 0, 1, 0, 0.444, 0.70, 0.83, 1.19, 0.96],
    'V': [4.2, 140.0, 0, 0, 0, 0.386, 0.50, 1.06, 1.70, 0.50],
    'W': [-0.9, 227.8, 0, 0, 1, 0.305, 0.31, 1.08, 1.37, 0.96],
    'Y': [-1.3, 193.6, 0, 1, 1, 0.420, 0.59, 0.69, 1.47, 1.14],
}


def get_residue_properties(residue_1letter):
    """
    Obtém propriedades físico-químicas de um resíduo.
    
    Args:
        residue_1letter: Código de 1 letra do aminoácido
    
    Returns:
        dict com 15 propriedades físico-químicas
    """
    if residue_1letter not in AMINO_ACID_PROPERTIES:
        # Resíduo não-padrão: usar valores neutros
        props = [0.0, 120.0, 0, 0, 0, 0.4, 0.6, 1.0, 1.0, 1.0]
    else:
        props = AMINO_ACID_PROPERTIES[residue_1letter]
    
    hydrophobicity, volume, charge, polarity, aromaticity, \
    flexibility, accessibility, helix_prop, sheet_prop, turn_prop = props
    
    # Indicadores categóricos
    is_gly = 1 if residue_1letter == 'G' else 0
    is_pro = 1 if residue_1letter == 'P' else 0
    is_charged = 1 if abs(charge) > 0.1 else 0
    is_aromatic = aromaticity
    is_aliphatic = 1 if residue_1letter in ['I', 'L', 'V', 'A'] else 0
    
    return {
        'hydrophobicity': hydrophobicity,
        'volume': volume,
        'charge': charge,
        'polarity': polarity,
        'aromaticity': aromaticity,
        'flexibility': flexibility,
        'accessibility': accessibility,
        'helix_propensity': helix_prop,
        'sheet_propensity': sheet_prop,
        'turn_propensity': turn_prop,
        'is_gly': is_gly,
        'is_pro': is_pro,
        'is_charged': is_charged,
        'is_aromatic': is_aromatic,
        'is_aliphatic': is_aliphatic,
    }


def calculate_features_single_structure(coords, sequence):
    """
    Calcula 27 features (12 geométricas + 15 físico-químicas) para uma única estrutura.
    
    Args:
        coords: np.ndarray (N, 3) - Coordenadas Cα
        sequence: str - Sequência de aminoácidos (1-letter codes)
    
    Returns:
        pd.DataFrame com 27 features
    """
    n_residues = len(coords)
    
    # Centroide
    centroid = coords.mean(axis=0)
    
    # Raio de giração
    R_g = np.sqrt(np.mean(np.sum((coords - centroid)**2, axis=1)))
    
    # Matriz de distâncias
    dist_matrix = distance_matrix(coords, coords)
    
    features = []
    
    for i in range(n_residues):
        residue_1letter = sequence[i]
        
        # ============================================================
        # FEATURES GEOMÉTRICAS (12)
        # ============================================================
        
        # k-nearest neighbors (k=5)
        distances_to_others = dist_matrix[i].copy()
        distances_to_others[i] = np.inf
        sorted_distances = np.sort(distances_to_others)
        
        d1 = sorted_distances[0]  # 1º vizinho mais próximo
        d2 = sorted_distances[1]  # 2º vizinho mais próximo
        d3 = sorted_distances[2]  # 3º vizinho mais próximo
        d4 = sorted_distances[3]  # 4º vizinho mais próximo
        d5 = sorted_distances[4]  # 5º vizinho mais próximo
        
        # Distância média a todos os resíduos
        valid_distances = distances_to_others[distances_to_others < np.inf]
        mean_dist = valid_distances.mean()
        
        # Distância ao centroide
        dist_to_center = np.linalg.norm(coords[i] - centroid)
        
        # Densidade local (Cα em raio de 8 Å)
        radius = 8.0
        local_density = (dist_matrix[i] <= radius).sum() - 1
        
        # Distância sequencial ao N-terminal
        dist_to_n_term = i
        
        # Distância sequencial ao C-terminal
        dist_to_c_term = n_residues - 1 - i
        
        # Distância máxima
        max_dist = valid_distances.max()
        
        # ============================================================
        # FEATURES FÍSICO-QUÍMICAS (15)
        # ============================================================
        
        res_props = get_residue_properties(residue_1letter)
        
        # Consolidar todas as features
        feature_row = {
            # Geométricas (12)
            'd1': d1,
            'd2': d2,
            'd3': d3,
            'd4': d4,
            'd5': d5,
            'R_g': R_g,
            'mean_dist': mean_dist,
            'dist_to_center': dist_to_center,
            'local_density': local_density,
            'dist_to_n_term': dist_to_n_term,
            'dist_to_c_term': dist_to_c_term,
            'max_dist': max_dist,
            # Físico-químicas (15)
            'hydrophobicity': res_props['hydrophobicity'],
            'volume': res_props['volume'],
            'charge': res_props['charge'],
            'polarity': res_props['polarity'],
            'aromaticity': res_props['aromaticity'],
            'flexibility': res_props['flexibility'],
            'accessibility': res_props['accessibility'],
            'helix_propensity': res_props['helix_propensity'],
            'sheet_propensity': res_props['sheet_propensity'],
            'turn_propensity': res_props['turn_propensity'],
            'is_gly': res_props['is_gly'],
            'is_pro': res_props['is_pro'],
            'is_charged': res_props['is_charged'],
            'is_aromatic': res_props['is_aromatic'],
            'is_aliphatic': res_props['is_aliphatic'],
        }
        
        features.append(feature_row)
    
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
    
    model_file = models_path / 'ridge_model_v2.pkl'
    scaler_file = models_path / 'scaler_v2.pkl'
    metadata_file = models_path / 'metadata_v2.json'
    
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
        print(f"MAE esperado (CV): {metadata.get('mae_cv', 'N/A'):.3f} ppm")
        print(f"Features: {metadata.get('n_features', 'N/A')}")
    
    return model, scaler, metadata


def predict_shifts(pdb_path, chain_id='A', models_dir='models', output_path=None):
    """
    Prediz chemical shifts CA para estrutura PDB usando 27 features.
    
    Args:
        pdb_path: Caminho para arquivo PDB
        chain_id: ID da cadeia a processar
        models_dir: Diretório com modelo treinado
        output_path: Caminho para salvar CSV (opcional)
    
    Returns:
        pd.DataFrame com predições
    """
    print(f"\n{'='*70}")
    print(f"PREDIÇÃO DE CHEMICAL SHIFTS ¹³C-α (27 FEATURES)")
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
    
    # 3. Calcular features (27: 12 geométricas + 15 físico-químicas)
    print("3. Calculando features (27 total)...")
    coords = structure.ca_coords
    sequence = structure.sequence_1letter
    features_df = calculate_features_single_structure(coords, sequence)
    print(f"   ✓ 12 features geométricas calculadas")
    print(f"   ✓ 15 features físico-químicas atribuídas")
    print(f"   ✓ Total: {len(features_df.columns)} features para {len(features_df)} resíduos\n")
    
    # 4. Verificar ordem das features
    expected_features = metadata.get('feature_names', None)
    
    if expected_features is None:
        # Ordem padrão se metadata não especificar
        expected_features = [
            # Geométricas (12)
            'd1', 'd2', 'd3', 'd4', 'd5',
            'R_g', 'mean_dist', 'dist_to_center',
            'local_density', 'dist_to_n_term', 'dist_to_c_term', 'max_dist',
            # Físico-químicas (15)
            'hydrophobicity', 'volume', 'charge', 'polarity', 'aromaticity',
            'flexibility', 'accessibility', 'helix_propensity', 
            'sheet_propensity', 'turn_propensity',
            'is_gly', 'is_pro', 'is_charged', 'is_aromatic', 'is_aliphatic'
        ]
    
    # Garantir ordem correta
    features_df = features_df[expected_features]
    
    # 5. Normalizar features
    print("4. Normalizando features (z-score)...")
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
        output_path = pdb_path.stem + '_shifts_predicted_v2.csv'
    
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
    
    # 11. Informações do modelo
    if metadata:
        print("Informações do Modelo:")
        print(f"  Versão: {metadata.get('version', 'N/A')}")
        print(f"  MAE (CV): {metadata.get('mae_cv', 'N/A'):.3f} ppm")
        print(f"  RMSE (CV): {metadata.get('rmse_cv', 'N/A'):.3f} ppm")
        print(f"  R² (CV): {metadata.get('r2_cv', 'N/A'):.3f}")
        print(f"  Lambda: {metadata.get('lambda_fullfit', 'N/A')}")
        print()
    
    return results


def main():
    """Função principal - interface CLI."""
    
    parser = argparse.ArgumentParser(
        description='Prediz chemical shifts ¹³C-α a partir de estrutura PDB (27 features)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python scripts/predict_shifts.py --pdb estrutura.pdb
  python scripts/predict_shifts.py --pdb estrutura.pdb --output resultados.csv
  python scripts/predict_shifts.py --pdb estrutura.pdb --chain B
  python scripts/predict_shifts.py --pdb estrutura.pdb --models models/

Features calculadas (27 total):
  - 12 geométricas: d1-d5, R_g, mean_dist, dist_to_center, local_density, 
                    dist_to_n_term, dist_to_c_term, max_dist
  - 15 físico-químicas: hydrophobicity, volume, charge, polarity, aromaticity,
                        flexibility, accessibility, helix/sheet/turn propensity,
                        is_gly, is_pro, is_charged, is_aromatic, is_aliphatic
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
        help='Arquivo CSV de saída (padrão: <pdb_name>_shifts_predicted_v2.csv)'
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
        print(f"Use o arquivo gerado para validação ou análise downstream.\n")
        
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