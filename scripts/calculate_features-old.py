# scripts/calculate_features.py

"""
Calcula features geométricas a partir das coordenadas Cα.

Features Fase 0 (mínimas):
- dist_to_center: distância euclidiana ao centroide da proteína
- dist_to_nearest_neighbor: distância ao Cα vizinho mais próximo
- local_density: número de Cα em raio de 8 Å
- dist_to_n_term: distância sequencial ao N-terminal
- dist_to_c_term: distância sequencial ao C-terminal

Input: data/processed/pilot_dataset_raw.csv
Output: data/processed/pilot_dataset_features.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix

def calculate_geometric_features(df):
    """
    Calcula features geométricas para o dataset.
    
    Args:
        df: DataFrame com colunas ca_x, ca_y, ca_z, pdb_id, residue_index
    
    Returns:
        DataFrame com features adicionadas
    """
    print(f"\n{'='*70}")
    print(f"CALCULANDO FEATURES GEOMÉTRICAS")
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
        
        # 1. Centroide da proteína
        centroid = coords.mean(axis=0)
        
        # 2. Matriz de distâncias (todas contra todas)
        dist_matrix = distance_matrix(coords, coords)
        
        # Para cada resíduo, calcular features
        for idx, row in structure_df.iterrows():
            i = idx  # Índice no array local desta estrutura
            
            # Feature 1: Distância ao centroide
            dist_to_center = np.linalg.norm(coords[i] - centroid)
            
            # Feature 2: Distância ao vizinho mais próximo
            # (excluir distância a si mesmo = 0)
            distances_to_others = dist_matrix[i].copy()
            distances_to_others[i] = np.inf  # Ignorar si mesmo
            dist_to_nearest = distances_to_others.min()
            
            # Feature 3: Densidade local (Cα em raio de 8 Å)
            radius = 8.0  # Angstroms
            within_radius = (dist_matrix[i] <= radius).sum() - 1  # -1 para excluir si mesmo
            local_density = within_radius
            
            # Feature 4: Distância ao N-terminal (índice sequencial)
            dist_to_n_term = i  # Resíduo 0 é N-terminal
            
            # Feature 5: Distância ao C-terminal (índice sequencial)
            dist_to_c_term = n_residues - 1 - i  # Último resíduo é C-terminal
            
            # Armazenar features
            feature_row = {
                'pdb_id': row['pdb_id'],
                'bmrb_id': row['bmrb_id'],
                'residue_index': row['residue_index'],
                'residue_type': row['residue_type'],
                'residue_type_3letter': row['residue_type_3letter'],
                'ca_x': row['ca_x'],
                'ca_y': row['ca_y'],
                'ca_z': row['ca_z'],
                'shift_ca_experimental': row['shift_ca_experimental'],
                # Features geométricas
                'dist_to_center': dist_to_center,
                'dist_to_nearest_neighbor': dist_to_nearest,
                'local_density': local_density,
                'dist_to_n_term': dist_to_n_term,
                'dist_to_c_term': dist_to_c_term,
            }
            
            feature_rows.append(feature_row)
        
        print(f"  ✓ Features calculadas para {n_residues} resíduos")
        print(f"    - Centro geométrico: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
        print(f"    - Range dist_to_center: {coords[:, 0].min():.2f} - {coords[:, 0].max():.2f} Å")
        print(f"    - Range local_density: {0} - {within_radius}")
    
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
    
    feature_cols = [
        'dist_to_center',
        'dist_to_nearest_neighbor', 
        'local_density',
        'dist_to_n_term',
        'dist_to_c_term'
    ]
    
    for col in feature_cols:
        print(f"{col}:")
        print(f"  Min: {features_df[col].min():.2f}")
        print(f"  Max: {features_df[col].max():.2f}")
        print(f"  Média: {features_df[col].mean():.2f}")
        print(f"  Std: {features_df[col].std():.2f}")
        print()
    
    # Salvar
    output_path = Path('data/processed/pilot_dataset_features.csv')
    features_df.to_csv(output_path, index=False)
    
    print(f"{'='*70}")
    print(f"  ✓ Dataset com features salvo em: {output_path}")
    print(f"  - Total de amostras: {len(features_df)}")
    print(f"  - Total de features: {len(feature_cols)}")
    print(f"{'='*70}\n")
    
    # Preview
    print("Preview (primeiras 5 linhas):")
    display_cols = ['pdb_id', 'residue_index', 'residue_type', 
                    'dist_to_center', 'local_density', 'shift_ca_experimental']
    print(features_df[display_cols].head().to_string())
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()