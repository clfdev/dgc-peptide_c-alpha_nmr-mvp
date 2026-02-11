# scripts/calculate_features_v2.py

"""
Calcula features geométricas e físico-químicas a partir das coordenadas Cα.

Features Fase 0 Expandidas (27 features):

GEOMÉTRICAS (12 features):
- d1, d2, d3, d4, d5: distâncias aos 5 vizinhos Cα mais próximos
- R_g: raio de giração (radius of gyration)
- mean_dist: distância média a todos os Cα
- dist_to_center: distância ao centroide da proteína
- local_density: número de Cα em raio de 8 Å
- dist_to_n_term: distância sequencial ao N-terminal
- dist_to_c_term: distância sequencial ao C-terminal
- max_dist: distância ao Cα mais distante

FÍSICO-QUÍMICAS (15 features):
- hydrophobicity: hidrofobicidade do resíduo (escala Kyte-Doolittle)
- volume: volume do resíduo (Å³)
- charge: carga formal em pH 7
- polarity: polaridade (polar=1, apolar=0)
- aromaticity: aromaticidade (aromático=1, não-aromático=0)
- flexibility: flexibilidade conformacional (escala Vihinen)
- accessibility: acessibilidade ao solvente relativa (escala Janin)
- helix_propensity: propensão a formar α-hélice
- sheet_propensity: propensão a formar β-folha
- turn_propensity: propensão a formar turns
- is_gly: indicador se é glicina (1=sim, 0=não)
- is_pro: indicador se é prolina (1=sim, 0=não)
- is_charged: indicador se é carregado (1=sim, 0=não)
- is_aromatic: indicador se é aromático (1=sim, 0=não)
- is_aliphatic: indicador se é alifático (1=sim, 0=não)

Input: data/processed/pilot_dataset_raw.csv
Output: data/processed/pilot_dataset_features_v2.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix

# Propriedades físico-químicas dos aminoácidos
AMINO_ACID_PROPERTIES = {
    # Formato: [hydrophobicity, volume, charge, polarity, aromaticity, 
    #           flexibility, accessibility, helix_prop, sheet_prop, turn_prop]
    'A': [1.8, 88.6, 0, 0, 0, 0.360, 0.62, 1.42, 0.83, 0.66],   # Alanine
    'C': [2.5, 108.5, 0, 0, 0, 0.346, 0.29, 0.70, 1.19, 1.19],  # Cysteine
    'D': [-3.5, 111.1, -1, 1, 0, 0.511, 0.90, 1.01, 0.54, 1.46], # Aspartate
    'E': [-3.5, 138.4, -1, 1, 0, 0.497, 0.75, 1.51, 0.37, 0.74], # Glutamate
    'F': [2.8, 189.9, 0, 0, 1, 0.314, 0.42, 1.13, 1.38, 0.60],  # Phenylalanine
    'G': [-0.4, 60.1, 0, 0, 0, 0.544, 0.81, 0.57, 0.75, 1.56],  # Glycine
    'H': [-3.2, 153.2, 0.5, 1, 1, 0.323, 0.67, 1.00, 0.87, 0.95], # Histidine
    'I': [4.5, 166.7, 0, 0, 0, 0.462, 0.46, 1.08, 1.60, 0.47],  # Isoleucine
    'K': [-3.9, 168.6, 1, 1, 0, 0.466, 0.85, 1.16, 0.74, 1.01],  # Lysine
    'L': [3.8, 166.7, 0, 0, 0, 0.365, 0.45, 1.21, 1.30, 0.59],  # Leucine
    'M': [1.9, 162.9, 0, 0, 0, 0.295, 0.40, 1.45, 1.05, 0.60],  # Methionine
    'N': [-3.5, 114.1, 0, 1, 0, 0.463, 0.78, 0.67, 0.89, 1.56],  # Asparagine
    'P': [-1.6, 112.7, 0, 0, 0, 0.509, 0.76, 0.57, 0.55, 1.52],  # Proline
    'Q': [-3.5, 143.8, 0, 1, 0, 0.493, 0.84, 1.11, 1.10, 0.98],  # Glutamine
    'R': [-4.5, 173.4, 1, 1, 0, 0.529, 0.95, 0.98, 0.93, 0.95],  # Arginine
    'S': [-0.8, 89.0, 0, 1, 0, 0.507, 0.78, 0.77, 0.75, 1.43],  # Serine
    'T': [-0.7, 116.1, 0, 1, 0, 0.444, 0.70, 0.83, 1.19, 0.96],  # Threonine
    'V': [4.2, 140.0, 0, 0, 0, 0.386, 0.50, 1.06, 1.70, 0.50],  # Valine
    'W': [-0.9, 227.8, 0, 0, 1, 0.305, 0.31, 1.08, 1.37, 0.96],  # Tryptophan
    'Y': [-1.3, 193.6, 0, 1, 1, 0.420, 0.59, 0.69, 1.47, 1.14],  # Tyrosine
}

def get_residue_properties(residue_1letter):
    """
    Obtém propriedades físico-químicas de um resíduo.
    
    Args:
        residue_1letter: Código de 1 letra do aminoácido
    
    Returns:
        dict com propriedades
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


def calculate_geometric_features(df):
    """
    Calcula features geométricas e físico-químicas para o dataset.
    
    Args:
        df: DataFrame com colunas ca_x, ca_y, ca_z, pdb_id, residue_index, residue_type
    
    Returns:
        DataFrame com 27 features adicionadas
    """
    print(f"\n{'='*70}")
    print(f"CALCULANDO FEATURES EXPANDIDAS (27 features)")
    print(f"{'='*70}\n")
    
    # Lista para armazenar features
    feature_rows = []
    
    # Processar cada estrutura separadamente
    for pdb_id in sorted(df['pdb_id'].unique()):
        print(f"Processando estrutura: {pdb_id}")
        
        # Filtrar dados desta estrutura
        structure_df = df[df['pdb_id'] == pdb_id].copy()
        structure_df = structure_df.sort_values('residue_index').reset_index(drop=True)
        
        n_residues = len(structure_df)
        print(f"  - {n_residues} resíduos")
        
        # Extrair coordenadas Cα como array Nx3
        coords = structure_df[['ca_x', 'ca_y', 'ca_z']].values
        
        # Centroide da proteína
        centroid = coords.mean(axis=0)
        
        # Raio de giração (Radius of Gyration)
        R_g = np.sqrt(np.mean(np.sum((coords - centroid)**2, axis=1)))
        
        # Matriz de distâncias (todas contra todas)
        dist_matrix = distance_matrix(coords, coords)
        
        # Para cada resíduo, calcular features
        for idx, row in structure_df.iterrows():
            i = idx  # Índice no array local desta estrutura
            
            residue_1letter = row['residue_type']
            
            # ============================================================
            # FEATURES GEOMÉTRICAS (12)
            # ============================================================
            
            # Distâncias para k-nearest neighbors (k=5)
            distances_to_others = dist_matrix[i].copy()
            distances_to_others[i] = np.inf  # Ignorar si mesmo
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
            radius = 8.0  # Angstroms
            local_density = (dist_matrix[i] <= radius).sum() - 1  # -1 para excluir si mesmo
            
            # Distância sequencial ao N-terminal
            dist_to_n_term = i  # Resíduo 0 é N-terminal
            
            # Distância sequencial ao C-terminal
            dist_to_c_term = n_residues - 1 - i  # Último resíduo é C-terminal
            
            # Distância máxima (ao Cα mais distante)
            max_dist = valid_distances.max()
            
            # ============================================================
            # FEATURES FÍSICO-QUÍMICAS (15)
            # ============================================================
            
            res_props = get_residue_properties(residue_1letter)
            
            # Construir dicionário de features completo
            feature_row = {
                # Metadados
                'pdb_id': row['pdb_id'],
                'bmrb_id': row['bmrb_id'],
                'residue_index': row['residue_index'],
                'residue_type': residue_1letter,
                'residue_type_3letter': row['residue_type_3letter'],
                'ca_x': row['ca_x'],
                'ca_y': row['ca_y'],
                'ca_z': row['ca_z'],
                'shift_ca_experimental': row['shift_ca_experimental'],
                
                # FEATURES GEOMÉTRICAS (12)
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
                
                # FEATURES FÍSICO-QUÍMICAS (15)
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
            
            feature_rows.append(feature_row)
        
        print(f"  ✓ 27 features calculadas para {n_residues} resíduos")
        print(f"    - Centro geométrico: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
        print(f"    - Raio de giração: {R_g:.2f} Å")
        print(f"    - Range d1: {sorted_distances[0]:.2f} Å")
        print(f"    - Range local_density: 0 - {int((dist_matrix < 8.0).sum(axis=1).max() - 1)}")
    
    # Criar DataFrame com features
    features_df = pd.DataFrame(feature_rows)
    
    return features_df


def main():
    """Função principal."""
    
    # Carregar dataset raw
    input_path = Path('data/processed/pilot_dataset_raw.csv')
    
    if not input_path.exists():
        print(f"Erro: {input_path} não encontrado!")
        print("Execute build_dataset.py primeiro.")
        sys.exit(1)
    
    print(f"Carregando dataset: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  - {len(df)} amostras carregadas")
    print(f"  - {df['pdb_id'].nunique()} estruturas")
    
    # Calcular features
    features_df = calculate_geometric_features(df)
    
    # Estatísticas das features
    print(f"\n{'='*70}")
    print(f"ESTATÍSTICAS DAS FEATURES")
    print(f"{'='*70}\n")
    
    # Features geométricas
    geometric_features = [
        'd1', 'd2', 'd3', 'd4', 'd5',
        'R_g', 'mean_dist', 'dist_to_center',
        'local_density', 'dist_to_n_term', 'dist_to_c_term', 'max_dist'
    ]
    
    print("FEATURES GEOMÉTRICAS (12):")
    for col in geometric_features:
        print(f"  {col:20s}: min={features_df[col].min():7.2f}, "
              f"max={features_df[col].max():7.2f}, "
              f"mean={features_df[col].mean():7.2f}, "
              f"std={features_df[col].std():7.2f}")
    
    # Features físico-químicas
    physicochemical_features = [
        'hydrophobicity', 'volume', 'charge', 'polarity', 'aromaticity',
        'flexibility', 'accessibility', 'helix_propensity', 
        'sheet_propensity', 'turn_propensity',
        'is_gly', 'is_pro', 'is_charged', 'is_aromatic', 'is_aliphatic'
    ]
    
    print("\nFEATURES FÍSICO-QUÍMICAS (15):")
    for col in physicochemical_features:
        print(f"  {col:20s}: min={features_df[col].min():7.2f}, "
              f"max={features_df[col].max():7.2f}, "
              f"mean={features_df[col].mean():7.2f}, "
              f"std={features_df[col].std():7.2f}")
    
    # Salvar
    output_path = Path('data/processed/pilot_dataset_features_v2.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"  ✓ Dataset com 27 features salvo em: {output_path}")
    print(f"  - Total de amostras: {len(features_df)}")
    print(f"  - Features geométricas: {len(geometric_features)}")
    print(f"  - Features físico-químicas: {len(physicochemical_features)}")
    print(f"  - Total de features: {len(geometric_features) + len(physicochemical_features)}")
    print(f"{'='*70}\n")
    
    # Preview
    print("Preview (primeiras 5 linhas - apenas features):")
    feature_cols = geometric_features[:5] + physicochemical_features[:3] + ['shift_ca_experimental']
    print(features_df[['pdb_id', 'residue_index', 'residue_type'] + feature_cols].head().to_string())
    
    # Distribuição de tipos de resíduos
    print(f"\n{'='*70}")
    print("DISTRIBUIÇÃO DE TIPOS DE RESÍDUOS:")
    print(f"{'='*70}\n")
    residue_counts = features_df['residue_type'].value_counts().sort_index()
    for res, count in residue_counts.items():
        percentage = (count / len(features_df)) * 100
        mean_shift = features_df[features_df['residue_type'] == res]['shift_ca_experimental'].mean()
        print(f"  {res}: {count:3d} ({percentage:5.1f}%) - shift médio: {mean_shift:.2f} ppm")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()