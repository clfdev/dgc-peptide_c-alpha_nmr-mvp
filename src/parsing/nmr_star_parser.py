# src/parsing/nmr_star_parser.py

"""
Parser para arquivos BMRB NMR-STAR (.str) - VERSÃO ROBUSTA.
Extrai sequência, chemical shifts CA (13C) e metadados.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BMRBEntity:
    """Entidade (sequência) do BMRB."""
    entity_id: int
    sequence_3letter: List[str]
    sequence_1letter: str
    n_residues: int
    
    def __str__(self):
        return (
            f"BMRBEntity(id={self.entity_id}, "
            f"n_residues={self.n_residues})"
        )


@dataclass
class ChemicalShift:
    """Chemical shift individual."""
    shift_id: int
    comp_index_id: int
    comp_id: str
    atom_id: str
    atom_type: str
    value: float
    value_error: Optional[float]
    ambiguity_code: int


@dataclass
class BMRBEntry:
    """Entrada BMRB parseada."""
    bmrb_id: str
    title: Optional[str]
    entities: List[BMRBEntity]
    ca_shifts: Dict[int, float]
    all_shifts: List[ChemicalShift]
    n_ca_shifts: int
    ca_coverage: float
    
    def __str__(self):
        return (
            f"BMRBEntry({self.bmrb_id}, "
            f"entities={len(self.entities)}, "
            f"ca_shifts={self.n_ca_shifts})"
        )


class NMRStarParser:
    """Parser de arquivos NMR-STAR (.str) - VERSÃO ROBUSTA."""
    
    AA_3_TO_1 = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        'MSE': 'M', 'CSO': 'C', 'MLY': 'K', 'PTR': 'Y',
        'SEP': 'S', 'TPO': 'T',
    }
    
    def __init__(self):
        pass
    
    def parse_bmrb_file(self, bmrb_path: str) -> BMRBEntry:
        """Parse arquivo BMRB NMR-STAR."""
        bmrb_path = Path(bmrb_path)
        
        match = re.search(r'bmr(\d+)', bmrb_path.name)
        bmrb_id = match.group(1) if match else bmrb_path.stem
        
        if not bmrb_path.exists():
            raise FileNotFoundError(f"Arquivo BMRB não encontrado: {bmrb_path}")
        
        with open(bmrb_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extrair componentes
        title = self._extract_title(content)
        entities = self._extract_entities_robust(content)
        all_shifts = self._extract_chemical_shifts_robust(content)
        
        # Filtrar CA shifts
        ca_shifts = {}
        for shift in all_shifts:
            if shift.atom_id == 'CA' and shift.atom_type == 'C':
                ca_shifts[shift.comp_index_id] = shift.value
        
        # Calcular cobertura
        if entities:
            n_residues = entities[0].n_residues
            ca_coverage = len(ca_shifts) / n_residues if n_residues > 0 else 0.0
        else:
            ca_coverage = 0.0
        
        return BMRBEntry(
            bmrb_id=bmrb_id,
            title=title,
            entities=entities,
            ca_shifts=ca_shifts,
            all_shifts=all_shifts,
            n_ca_shifts=len(ca_shifts),
            ca_coverage=ca_coverage
        )
    
    def _extract_title(self, content: str) -> Optional[str]:
        """Extrai título."""
        # Padrão 1: _Entry.Title
        match = re.search(r'_Entry\.Title\s+;([^;]+);', content, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Padrão 2: Com aspas
        match = re.search(r'_Entry\.Title\s+[\'"]([^\'"]+)[\'"]', content)
        if match:
            return match.group(1).strip()
        
        return None
    
    def _extract_entities_robust(self, content: str) -> List[BMRBEntity]:
        """
        Extrai entidades de forma robusta.
        Busca por diferentes padrões de loop.
        """
        entities = []
        
        # Padrão: loop_ seguido de _Entity_comp_index ou _Comp_index
        patterns = [
            r'loop_\s+(_Entity_comp_index\.\w+\s+)+\s*([\d\s\w\.\'"]+)\s*stop_',
            r'loop_\s+(_Comp_index\.\w+\s+)+\s*([\d\s\w\.\'"]+)\s*stop_',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.DOTALL)
            
            for match in matches:
                # Extrair cabeçalhos
                headers_section = match.group(0)
                headers = re.findall(r'_Entity_comp_index\.(\w+)', headers_section)
                
                if not headers:
                    continue
                
                logger.info(f"Encontrado loop _Entity_comp_index com colunas: {headers}")
                
                # Encontrar índices
                try:
                    comp_id_idx = headers.index('Comp_ID')
                except ValueError:
                    logger.warning("Comp_ID não encontrado")
                    continue
                
                # Entity_ID pode estar ausente
                entity_id_idx = headers.index('Entity_ID') if 'Entity_ID' in headers else None
                
                # Extrair dados (texto entre cabeçalhos e stop_)
                data_section = match.group(2) if len(match.groups()) > 1 else match.group(0)
                
                # Parse linhas
                lines = [l.strip() for l in data_section.split('\n') if l.strip() and not l.strip().startswith('_')]
                
                entities_dict = {}
                
                for line in lines:
                    if line == 'stop_' or line.startswith('save_') or line.startswith('loop_'):
                        break
                    
                    tokens = line.split()
                    if len(tokens) < len(headers):
                        continue
                    
                    entity_id = int(tokens[entity_id_idx]) if entity_id_idx is not None else 1
                    comp_id = tokens[comp_id_idx].strip('"\'')
                    
                    if entity_id not in entities_dict:
                        entities_dict[entity_id] = []
                    
                    entities_dict[entity_id].append(comp_id)
                
                # Criar BMRBEntity
                for entity_id, sequence_3letter in entities_dict.items():
                    sequence_1letter = ''.join(
                        self.AA_3_TO_1.get(aa, 'X') for aa in sequence_3letter
                    )
                    
                    entity = BMRBEntity(
                        entity_id=entity_id,
                        sequence_3letter=sequence_3letter,
                        sequence_1letter=sequence_1letter,
                        n_residues=len(sequence_3letter)
                    )
                    
                    entities.append(entity)
                    logger.info(f"Entidade {entity_id} extraída: {len(sequence_3letter)} resíduos")
        
        return entities
    
    def _extract_chemical_shifts_robust(self, content: str) -> List[ChemicalShift]:
        """
        Extrai chemical shifts de forma robusta.
        """
        shifts = []
        
        # Procurar loop _Atom_chem_shift
        pattern = r'loop_\s+(_Atom_chem_shift\.\w+\s+)+(.*?)stop_'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            full_match = match.group(0)
            
            # Extrair cabeçalhos
            headers = re.findall(r'_Atom_chem_shift\.(\w+)', full_match)
            
            if not headers:
                continue
            
            logger.info(f"Encontrado loop _Atom_chem_shift com {len(headers)} colunas")
            
            # Mapear índices de colunas
            col_map = {header: idx for idx, header in enumerate(headers)}
            
            # Colunas essenciais
            required = ['Comp_index_ID', 'Comp_ID', 'Atom_ID', 'Atom_type', 'Val']
            if not all(col in col_map for col in required):
                missing = [col for col in required if col not in col_map]
                logger.warning(f"Colunas faltando: {missing}")
                continue
            
            # Extrair seção de dados
            data_section = match.group(2)
            lines = [l.strip() for l in data_section.split('\n') 
                    if l.strip() and not l.strip().startswith('_')]
            
            for line in lines:
                if line == 'stop_' or line.startswith('save_'):
                    break
                
                tokens = line.split()
                if len(tokens) < len(headers):
                    continue
                
                try:
                    shift_id = int(tokens[col_map.get('ID', 0)])
                    comp_index_id = int(tokens[col_map['Comp_index_ID']])
                    comp_id = tokens[col_map['Comp_ID']].strip('"\'')
                    atom_id = tokens[col_map['Atom_ID']].strip('"\'')
                    atom_type = tokens[col_map['Atom_type']].strip('"\'')
                    
                    val_str = tokens[col_map['Val']]
                    if val_str in ['.', '?', 'NA']:
                        continue
                    value = float(val_str)
                    
                    # Opcionais
                    value_error = None
                    if 'Val_err' in col_map:
                        err_str = tokens[col_map['Val_err']]
                        if err_str not in ['.', '?', 'NA']:
                            value_error = float(err_str)
                    
                    ambiguity_code = 1
                    if 'Ambiguity_code' in col_map:
                        amb_str = tokens[col_map['Ambiguity_code']]
                        if amb_str not in ['.', '?']:
                            ambiguity_code = int(amb_str)
                    
                    shift = ChemicalShift(
                        shift_id=shift_id,
                        comp_index_id=comp_index_id,
                        comp_id=comp_id,
                        atom_id=atom_id,
                        atom_type=atom_type,
                        value=value,
                        value_error=value_error,
                        ambiguity_code=ambiguity_code
                    )
                    
                    shifts.append(shift)
                
                except (ValueError, IndexError, KeyError):
                    continue
        
        logger.info(f"Total de shifts extraídos: {len(shifts)}")
        
        # Filtrar CA para log
        ca_count = sum(1 for s in shifts if s.atom_id == 'CA' and s.atom_type == 'C')
        logger.info(f"Shifts CA (13C): {ca_count}")
        
        return shifts


# ==================== FUNÇÃO DE TESTE ====================

def test_nmr_star_parser():
    """Testa o parser."""
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python nmr_star_parser.py <arquivo.str>")
        sys.exit(1)
    
    bmrb_file = sys.argv[1]
    
    parser = NMRStarParser()
    entry = parser.parse_bmrb_file(bmrb_file)
    
    print(f"\n{entry}")
    
    if entry.title:
        print(f"\nTítulo: {entry.title[:100]}...")
    
    print(f"\nEntidades ({len(entry.entities)}):")
    for entity in entry.entities:
        print(f"  {entity}")
        print(f"  Sequência: {entity.sequence_1letter}")
    
    print(f"\nChemical Shifts CA:")
    print(f"  - Total: {entry.n_ca_shifts}")
    print(f"  - Cobertura: {entry.ca_coverage*100:.1f}%")
    
    if entry.ca_shifts:
        print(f"\nPrimeiros 10 shifts CA:")
        for idx, (res_idx, shift) in enumerate(sorted(entry.ca_shifts.items())[:10]):
            entity = entry.entities[0] if entry.entities else None
            res_type = '???'
            if entity and 1 <= res_idx <= len(entity.sequence_3letter):
                res_type = entity.sequence_3letter[res_idx-1]
            print(f"  Resíduo {res_idx} ({res_type}): {shift:.2f} ppm")
    
    print(f"\nTotal de shifts (todos átomos): {len(entry.all_shifts)}")


if __name__ == "__main__":
    test_nmr_star_parser()