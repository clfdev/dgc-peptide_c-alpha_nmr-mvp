# scripts/build_dataset.py

"""
Constrói dataset combinando dados PDB + BMRB.

Para cada par:
1. Parse PDB → sequência + coordenadas Cα
2. Parse BMRB → shifts CA
3. Alinhamento de sequências
4. Gerar dataset: resíduo | coords | shift_experimental
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from parsing.pdb_parser import PDBParser
from parsing.nmr_star_parser import NMRStarParser
import yaml


def build_dataset_for_pair(pdb_path, bmrb_path, pdb_id, bmrb_id):
    """
    Constrói dataset para um par PDB-BMRB.
    
    Returns:
        DataFrame com colunas: pdb_id, bmrb_id, residue_index, residue_type,
                               ca_x, ca_y, ca_z, shift_ca_experimental
    """
    print(f"\n{'='*60}")
    print(f"Processando: {pdb_id} ↔ BMRB {bmrb_id}")
    print(f"{'='*60}")
    
    # Parse PDB
    print(f"  Parseando PDB {pdb_id}...")
    pdb_parser = PDBParser()
    pdb_structure = pdb_parser.parse_pdb_file(pdb_path, chain_id='A')
    
    print(f"    ✓ {pdb_structure.n_residues} resíduos")
    print(f"    ✓ Sequência: {pdb_structure.sequence_1letter}")
    
    # Parse BMRB
    print(f"  Parseando BMRB {bmrb_id}...")
    bmrb_parser = NMRStarParser()
    bmrb_entry = bmrb_parser.parse_bmrb_file(bmrb_path)
    
    print(f"    ✓ {bmrb_entry.n_ca_shifts} shifts CA")
    print(f"    ✓ Cobertura: {bmrb_entry.ca_coverage*100:.1f}%")
    
    # Verificar se há sequência no BMRB
    if bmrb_entry.entities:
        bmrb_seq = bmrb_entry.entities[0].sequence_1letter
        print(f"    ✓ Sequência BMRB: {bmrb_seq}")
        
        # Verificar correspondência
        if pdb_structure.sequence_1letter == bmrb_seq:
            print(f"    ✓ Sequências PDB e BMRB são IDÊNTICAS")
            alignment_ok = True
        elif pdb_structure.sequence_1letter in bmrb_seq or bmrb_seq in pdb_structure.sequence_1letter:
            print(f"    ⚠ Sequências SIMILARES (diferença de comprimento)")
            print(f"      PDB:  {pdb_structure.sequence_1letter}")
            print(f"      BMRB: {bmrb_seq}")
            alignment_ok = True
        else:
            print(f"    ⚠ Sequências DIFEREM!")
            print(f"      PDB:  {pdb_structure.sequence_1letter}")
            print(f"      BMRB: {bmrb_seq}")
            alignment_ok = False
    else:
        print(f"    ⚠ BMRB sem sequência parseada (assumindo mapeamento direto)")
        alignment_ok = True  # Assumir que índices correspondem
    
    # Construir dataset
    print(f"  Construindo dataset...")
    data_rows = []
    
    for residue in pdb_structure.residues:
        res_idx = residue.index
        
        # Buscar shift CA correspondente
        shift_ca = bmrb_entry.ca_shifts.get(res_idx)
        
        if shift_ca is None:
            print(f"    ⚠ Resíduo {res_idx} ({residue.residue_1letter}) sem shift CA - pulando")
            continue
        
        row = {
            'pdb_id': pdb_id,
            'bmrb_id': bmrb_id,
            'residue_index': res_idx,
            'residue_type': residue.residue_1letter,
            'residue_type_3letter': residue.residue_type,
            'ca_x': residue.ca_coords[0],
            'ca_y': residue.ca_coords[1],
            'ca_z': residue.ca_coords[2],
            'shift_ca_experimental': shift_ca,
        }
        
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    
    print(f"    ✓ {len(df)} amostras geradas")
    if len(df) > 0:
        print(f"    ✓ Range de shifts: {df['shift_ca_experimental'].min():.2f} - {df['shift_ca_experimental'].max():.2f} ppm")
    
    return df


def build_full_dataset():
    """Constrói dataset completo para todas as entradas da lista piloto."""
    
    # Carregar pilot_list.yaml
    pilot_list_path = Path('data/pilot_list.yaml')
    
    if not pilot_list_path.exists():
        print(f"Erro: {pilot_list_path} não encontrado!")
        print("Execute este script do diretório raiz do projeto.")
        sys.exit(1)
    
    with open(pilot_list_path, 'r') as f:
        config = yaml.safe_load(f)
    
    entries = config['entries']
    
    print(f"\n{'='*70}")
    print(f"CONSTRUINDO DATASET COMPLETO - {len(entries)} ENTRADAS")
    print(f"{'='*70}")
    
    all_data = []
    successful = 0
    failed = 0
    
    for entry in entries:
        pdb_id = entry['pdb_id']
        bmrb_id = entry['bmrb_id']
        
        pdb_path = f"data/raw/pdb/{pdb_id}.pdb"
        bmrb_path = f"data/raw/bmrb/bmr{bmrb_id}_3.str"
        
        # Verificar se arquivos existem
        if not Path(pdb_path).exists():
            print(f"\n  ✗ Arquivo PDB não encontrado: {pdb_path}")
            failed += 1
            continue
        
        if not Path(bmrb_path).exists():
            print(f"\n  ✗ Arquivo BMRB não encontrado: {bmrb_path}")
            failed += 1
            continue
        
        try:
            df = build_dataset_for_pair(pdb_path, bmrb_path, pdb_id, bmrb_id)
            
            if len(df) > 0:
                all_data.append(df)
                successful += 1
            else:
                print(f"  ✗ Nenhuma amostra gerada para {pdb_id}")
                failed += 1
                
        except Exception as e:
            print(f"\n  ✗ ERRO ao processar {pdb_id}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue
    
    # Concatenar todos os dados
    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        
        print(f"\n{'='*70}")
        print(f"DATASET FINAL")
        print(f"{'='*70}")
        print(f"  Estruturas processadas com sucesso: {successful}/{len(entries)}")
        print(f"  Estruturas com falha: {failed}/{len(entries)}")
        print(f"  Total de amostras: {len(full_df)}")
        print(f"  Estruturas únicas: {full_df['pdb_id'].nunique()}")
        print(f"  Tipos de resíduos: {full_df['residue_type'].nunique()}")
        print(f"  Range de shifts CA: {full_df['shift_ca_experimental'].min():.2f} - {full_df['shift_ca_experimental'].max():.2f} ppm")
        print(f"  Média de shifts CA: {full_df['shift_ca_experimental'].mean():.2f} ppm")
        print(f"  Desvio padrão: {full_df['shift_ca_experimental'].std():.2f} ppm")
        
        # Estatísticas por estrutura
        print(f"\n  Amostras por estrutura:")
        for pdb_id in sorted(full_df['pdb_id'].unique()):
            n_samples = len(full_df[full_df['pdb_id'] == pdb_id])
            print(f"    {pdb_id}: {n_samples} amostras")
        
        # Salvar
        output_path = Path('data/processed/pilot_dataset_raw.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_csv(output_path, index=False)
        
        print(f"\n  ✓ Dataset salvo em: {output_path}")
        
        # Mostrar preview
        print(f"\nPreview (primeiras 10 linhas):")
        print(full_df.head(10).to_string())
        
        print(f"\n{'='*70}\n")
        
        return full_df
    else:
        print(f"\n{'='*70}")
        print(f"  ✗ Nenhum dado gerado!")
        print(f"  Estruturas processadas: 0/{len(entries)}")
        print(f"  Estruturas com falha: {failed}/{len(entries)}")
        print(f"{'='*70}\n")
        return None


if __name__ == "__main__":
    build_full_dataset()