# src/parsing/pdb_parser.py

"""
Parser para arquivos PDB.
Extrai coordenadas Cα, sequência e informações de resíduos.

Suporta:
- Arquivos PDB padrão
- Múltiplos modelos (usa MODEL 1)
- Altlocs (usa 'A' ou maior ocupância)
- Insertion codes (renumera sequencialmente)
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Residue:
    """Informações de um resíduo."""
    index: int           # Índice sequencial (1, 2, 3, ...)
    pdb_number: int      # Número no PDB (pode ter gaps)
    pdb_icode: str       # Insertion code (geralmente vazio)
    residue_type: str    # Código 3 letras (ALA, GLY, ...)
    residue_1letter: str # Código 1 letra (A, G, ...)
    chain_id: str        # ID da cadeia (geralmente 'A')
    ca_coords: Tuple[float, float, float]  # (x, y, z)
    occupancy: float     # Ocupância
    altloc: str          # Localização alternativa


@dataclass
class PDBStructure:
    """Estrutura PDB parseada."""
    pdb_id: str
    chain_id: str
    residues: List[Residue]
    sequence_3letter: List[str]
    sequence_1letter: str
    ca_coords: np.ndarray  # Array Nx3 de coordenadas
    n_residues: int
    
    def __str__(self):
        return (
            f"PDBStructure({self.pdb_id}, chain={self.chain_id}, "
            f"n_residues={self.n_residues})"
        )


class PDBParser:
    """Parser de arquivos PDB."""
    
    # Mapeamento de códigos 3→1 letra
    AA_3_TO_1 = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        # Não-padrão (converter para equivalente)
        'MSE': 'M',  # Selenometionina → Metionina
        'CSO': 'C',  # S-hidroxicisteína → Cisteína
        'MLY': 'K',  # N-dimetil-lisina → Lisina
        'PTR': 'Y',  # O-fosfo-L-tirosina → Tirosina
        'SEP': 'S',  # Fosfo-serina → Serina
        'TPO': 'T',  # Fosfo-treonina → Treonina
    }
    
    def __init__(self):
        """Inicializa o parser."""
        pass
    
    def parse_pdb_file(
        self, 
        pdb_path: str, 
        chain_id: str = 'A',
        model: int = 1
    ) -> PDBStructure:
        """
        Parse arquivo PDB e extrai informações.
        
        Args:
            pdb_path: Caminho para arquivo .pdb
            chain_id: ID da cadeia a extrair (default: 'A')
            model: Número do modelo (para NMR, default: 1)
        
        Returns:
            PDBStructure com dados extraídos
        """
        pdb_path = Path(pdb_path)
        pdb_id = pdb_path.stem.upper()
        
        if not pdb_path.exists():
            raise FileNotFoundError(f"Arquivo PDB não encontrado: {pdb_path}")
        
        # Ler arquivo
        with open(pdb_path, 'r') as f:
            lines = f.readlines()
        
        # Extrair átomos CA
        ca_atoms = self._extract_ca_atoms(lines, chain_id, model)
        
        if not ca_atoms:
            raise ValueError(
                f"Nenhum átomo CA encontrado para cadeia {chain_id} "
                f"no modelo {model}"
            )
        
        # Processar resíduos
        residues = self._process_residues(ca_atoms)
        
        # Construir sequências
        sequence_3letter = [r.residue_type for r in residues]
        sequence_1letter = ''.join(r.residue_1letter for r in residues)
        
        # Array de coordenadas
        ca_coords = np.array([r.ca_coords for r in residues])
        
        return PDBStructure(
            pdb_id=pdb_id,
            chain_id=chain_id,
            residues=residues,
            sequence_3letter=sequence_3letter,
            sequence_1letter=sequence_1letter,
            ca_coords=ca_coords,
            n_residues=len(residues)
        )
    
    def _extract_ca_atoms(
        self, 
        lines: List[str], 
        chain_id: str,
        model: int
    ) -> List[Dict]:
        """
        Extrai linhas ATOM/HETATM de átomos CA.
        
        Args:
            lines: Linhas do arquivo PDB
            chain_id: ID da cadeia
            model: Número do modelo
        
        Returns:
            Lista de dicionários com informações dos átomos CA
        """
        ca_atoms = []
        current_model = 1
        in_model = True
        
        for line in lines:
            # Controle de modelos (para estruturas NMR)
            if line.startswith('MODEL'):
                current_model = int(line.split()[1])
                in_model = (current_model == model)
                continue
            
            if line.startswith('ENDMDL'):
                if current_model == model:
                    break  # Terminou o modelo desejado
                continue
            
            # Processar apenas linhas ATOM/HETATM do modelo correto
            if not in_model:
                continue
            
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            
            # Parse linha PDB (formato fixo)
            try:
                atom_name = line[12:16].strip()
                altloc = line[16].strip()
                res_name = line[17:20].strip()
                chain = line[21].strip()
                res_num = int(line[22:26].strip())
                icode = line[26].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                occupancy = float(line[54:60].strip()) if line[54:60].strip() else 1.0
                
            except (ValueError, IndexError) as e:
                # Linha mal formatada, pular
                continue
            
            # Filtrar: apenas CA da cadeia especificada
            if atom_name != 'CA':
                continue
            
            if chain != chain_id:
                continue
            
            ca_atoms.append({
                'res_name': res_name,
                'res_num': res_num,
                'icode': icode,
                'chain': chain,
                'altloc': altloc,
                'coords': (x, y, z),
                'occupancy': occupancy,
            })
        
        return ca_atoms
    
    def _process_residues(self, ca_atoms: List[Dict]) -> List[Residue]:
        """
        Processa átomos CA em objetos Residue.
        
        Lida com:
        - Altlocs (escolhe 'A' ou maior ocupância)
        - Insertion codes
        - Renumeração sequencial
        
        Args:
            ca_atoms: Lista de átomos CA extraídos
        
        Returns:
            Lista de objetos Residue
        """
        residues = []
        
        # Agrupar por (res_num, icode)
        grouped = {}
        for atom in ca_atoms:
            key = (atom['res_num'], atom['icode'])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(atom)
        
        # Processar cada resíduo
        seq_index = 1
        for (res_num, icode), atoms in sorted(grouped.items()):
            # Se múltiplos altlocs, escolher
            if len(atoms) > 1:
                # Preferir altloc 'A', senão maior ocupância
                altloc_a = [a for a in atoms if a['altloc'] == 'A']
                if altloc_a:
                    atom = altloc_a[0]
                else:
                    atom = max(atoms, key=lambda x: x['occupancy'])
            else:
                atom = atoms[0]
            
            # Converter 3→1 letra
            res_3letter = atom['res_name']
            res_1letter = self.AA_3_TO_1.get(res_3letter, 'X')
            
            if res_1letter == 'X':
                print(f"Aviso: Resíduo não-padrão {res_3letter} → X")
            
            residue = Residue(
                index=seq_index,
                pdb_number=res_num,
                pdb_icode=icode,
                residue_type=res_3letter,
                residue_1letter=res_1letter,
                chain_id=atom['chain'],
                ca_coords=atom['coords'],
                occupancy=atom['occupancy'],
                altloc=atom['altloc']
            )
            
            residues.append(residue)
            seq_index += 1
        
        return residues


# ==================== FUNÇÃO DE TESTE ====================

def test_pdb_parser():
    """Testa o parser em um arquivo PDB."""
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python pdb_parser.py <arquivo.pdb>")
        sys.exit(1)
    
    pdb_file = sys.argv[1]
    
    parser = PDBParser()
    structure = parser.parse_pdb_file(pdb_file)
    
    print(f"\n{structure}")
    print(f"\nSequência ({structure.n_residues} resíduos):")
    print(structure.sequence_1letter)
    
    print(f"\nPrimeiros 5 resíduos:")
    for res in structure.residues[:5]:
        print(f"  {res.index}: {res.residue_type} ({res.residue_1letter}) "
              f"@ {res.ca_coords}")
    
    print(f"\nCoordenadas Cα shape: {structure.ca_coords.shape}")
    print(f"Centro geométrico: {structure.ca_coords.mean(axis=0)}")


if __name__ == "__main__":
    test_pdb_parser()