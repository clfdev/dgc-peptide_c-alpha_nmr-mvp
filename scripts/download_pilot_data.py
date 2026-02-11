#!/usr/bin/env python3
# scripts/download_pilot_data.py

"""
Download automático de estruturas PDB e arquivos BMRB para lista piloto.

Funcionalidades:
- Lê pilot_list.yaml
- Download com retry automático
- Validação de integridade
- Cache (não re-baixa arquivos existentes)
- Relatório detalhado

Uso:
    python scripts/download_pilot_data.py
    python scripts/download_pilot_data.py --force  # Re-baixar tudo
    python scripts/download_pilot_data.py --only-pdb  # Apenas PDB
    python scripts/download_pilot_data.py --only-bmrb  # Apenas BMRB
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import requests
import yaml

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PilotDataDownloader:
    """Downloader para dados da lista piloto."""
    
    def __init__(self, pilot_list_path: str = "data/pilot_list.yaml"):
        """
        Inicializa o downloader.
        
        Args:
            pilot_list_path: Caminho para pilot_list.yaml
        """
        self.pilot_list_path = Path(pilot_list_path)
        self.config = self._load_config()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.config.get('download_config', {}).get(
                'user_agent', 'DGC-NMR-MVP/0.1'
            )
        })
        
        # Estatísticas
        self.stats = {
            'pdb_downloaded': 0,
            'pdb_cached': 0,
            'pdb_failed': 0,
            'bmrb_downloaded': 0,
            'bmrb_cached': 0,
            'bmrb_failed': 0,
        }
    
    def _load_config(self) -> Dict:
        """Carrega configuração do pilot_list.yaml."""
        if not self.pilot_list_path.exists():
            raise FileNotFoundError(
                f"pilot_list.yaml não encontrado em {self.pilot_list_path}"
            )
        
        with open(self.pilot_list_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuração carregada: {config['metadata']['total_entries']} entradas")
        return config
    
    def _create_directories(self) -> None:
        """Cria estrutura de diretórios necessária."""
        dirs = self.config.get('directory_structure', {})
        
        for dir_key, dir_path in dirs.items():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Diretório criado/verificado: {dir_path}")
    
    def _download_with_retry(
        self, 
        url: str, 
        output_path: Path,
        file_type: str = "file"
    ) -> bool:
        """
        Download com retry automático.
        
        Args:
            url: URL para download
            output_path: Caminho de saída
            file_type: Tipo de arquivo (para log)
        
        Returns:
            True se sucesso, False se falha
        """
        config = self.config.get('download_config', {})
        timeout = config.get('timeout', 60)
        max_attempts = config.get('retry_attempts', 3)
        retry_delay = config.get('retry_delay', 5)
        
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"Baixando {file_type} de {url} (tentativa {attempt}/{max_attempts})")
                
                response = self.session.get(url, timeout=timeout, stream=True)
                
                if response.status_code == 200:
                    # Escrever arquivo
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Validar tamanho mínimo
                    file_size = output_path.stat().st_size
                    if file_size < 100:  # Menor que 100 bytes = provavelmente erro
                        logger.warning(f"Arquivo muito pequeno ({file_size} bytes), possível erro")
                        output_path.unlink()  # Deletar arquivo inválido
                        raise ValueError(f"Arquivo inválido (tamanho: {file_size} bytes)")
                    
                    logger.info(f"✓ Download concluído: {output_path.name} ({file_size:,} bytes)")
                    return True
                
                elif response.status_code == 404:
                    logger.error(f"✗ Arquivo não encontrado (404): {url}")
                    return False
                
                else:
                    logger.warning(f"HTTP {response.status_code} em {url}")
                    if attempt < max_attempts:
                        logger.info(f"Aguardando {retry_delay}s antes de tentar novamente...")
                        time.sleep(retry_delay)
            
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout ao baixar de {url}")
                if attempt < max_attempts:
                    time.sleep(retry_delay)
            
            except Exception as e:
                logger.error(f"Erro ao baixar de {url}: {str(e)}")
                if attempt < max_attempts:
                    time.sleep(retry_delay)
        
        logger.error(f"✗ Falha após {max_attempts} tentativas: {url}")
        return False
    
    def download_pdb(self, pdb_id: str, force: bool = False) -> bool:
        """
        Download de arquivo PDB.
        
        Args:
            pdb_id: Código PDB (ex: '1LE1')
            force: Se True, re-baixa mesmo se existir
        
        Returns:
            True se sucesso, False se falha
        """
        pdb_dir = Path(self.config['directory_structure']['raw_pdb'])
        output_path = pdb_dir / f"{pdb_id}.pdb"
        
        # Verificar cache
        if output_path.exists() and not force:
            file_size = output_path.stat().st_size
            logger.info(f"○ Cache: {pdb_id}.pdb ({file_size:,} bytes)")
            self.stats['pdb_cached'] += 1
            return True
        
        # Download
        url_template = self.config['download_urls']['pdb_template']
        url = url_template.format(pdb_id=pdb_id)
        
        success = self._download_with_retry(url, output_path, f"PDB {pdb_id}")
        
        if success:
            self.stats['pdb_downloaded'] += 1
        else:
            self.stats['pdb_failed'] += 1
        
        return success
    
    def download_bmrb(self, bmrb_id: str, force: bool = False) -> bool:
        """
        Download de arquivo BMRB NMR-STAR.
        
        Args:
            bmrb_id: ID BMRB (ex: '34305')
            force: Se True, re-baixa mesmo se existir
        
        Returns:
            True se sucesso, False se falha
        """
        bmrb_dir = Path(self.config['directory_structure']['raw_bmrb'])
        output_path = bmrb_dir / f"bmr{bmrb_id}_3.str"
        
        # Verificar cache
        if output_path.exists() and not force:
            file_size = output_path.stat().st_size
            logger.info(f"○ Cache: bmr{bmrb_id}_3.str ({file_size:,} bytes)")
            self.stats['bmrb_cached'] += 1
            return True
        
        # Tentar ambos os templates de URL
        url_templates = [
            self.config['download_urls']['bmrb_template_b'],
            self.config['download_urls']['bmrb_template_a'],
        ]
        
        for i, url_template in enumerate(url_templates, 1):
            url = url_template.format(bmrb_id=bmrb_id)
            logger.info(f"Tentando URL {i}/2 para BMRB {bmrb_id}")
            
            success = self._download_with_retry(
                url, output_path, f"BMRB {bmrb_id}"
            )
            
            if success:
                self.stats['bmrb_downloaded'] += 1
                return True
        
        # Falhou em ambas URLs
        logger.error(f"✗ BMRB {bmrb_id} não encontrado em nenhuma URL")
        self.stats['bmrb_failed'] += 1
        return False
    
    def download_all(
        self, 
        force: bool = False,
        only_pdb: bool = False,
        only_bmrb: bool = False
    ) -> Tuple[int, int]:
        """
        Download de todas as entradas da lista piloto.
        
        Args:
            force: Re-baixar arquivos existentes
            only_pdb: Baixar apenas PDBs
            only_bmrb: Baixar apenas BMRBs
        
        Returns:
            Tupla (sucessos, falhas)
        """
        self._create_directories()
        
        entries = self.config.get('entries', [])
        total = len(entries)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"DOWNLOAD DA LISTA PILOTO - {total} ENTRADAS")
        logger.info(f"{'='*70}\n")
        
        success_count = 0
        fail_count = 0
        
        for i, entry in enumerate(entries, 1):
            pdb_id = entry['pdb_id']
            bmrb_id = entry['bmrb_id']
            
            logger.info(f"\n[{i}/{total}] {pdb_id} ↔ BMRB {bmrb_id}")
            logger.info("-" * 50)
            
            entry_success = True
            
            # Download PDB
            if not only_bmrb:
                if not self.download_pdb(pdb_id, force):
                    entry_success = False
            
            # Download BMRB
            if not only_pdb:
                if not self.download_bmrb(bmrb_id, force):
                    entry_success = False
            
            if entry_success:
                success_count += 1
            else:
                fail_count += 1
        
        return success_count, fail_count
    
    def print_summary(self) -> None:
        """Imprime sumário do download."""
        logger.info(f"\n{'='*70}")
        logger.info("SUMÁRIO DO DOWNLOAD")
        logger.info(f"{'='*70}\n")
        
        logger.info("PDB:")
        logger.info(f"  - Baixados: {self.stats['pdb_downloaded']}")
        logger.info(f"  - Cache: {self.stats['pdb_cached']}")
        logger.info(f"  - Falhas: {self.stats['pdb_failed']}")
        
        logger.info("\nBMRB:")
        logger.info(f"  - Baixados: {self.stats['bmrb_downloaded']}")
        logger.info(f"  - Cache: {self.stats['bmrb_cached']}")
        logger.info(f"  - Falhas: {self.stats['bmrb_failed']}")
        
        total_downloaded = (
            self.stats['pdb_downloaded'] + self.stats['bmrb_downloaded']
        )
        total_cached = self.stats['pdb_cached'] + self.stats['bmrb_cached']
        total_failed = self.stats['pdb_failed'] + self.stats['bmrb_failed']
        
        logger.info(f"\nTOTAL:")
        logger.info(f"  - Arquivos baixados: {total_downloaded}")
        logger.info(f"  - Arquivos em cache: {total_cached}")
        logger.info(f"  - Falhas: {total_failed}")
        
        if total_failed == 0:
            logger.info(f"\n✓ Todos os downloads concluídos com sucesso!")
        else:
            logger.warning(f"\n⚠ {total_failed} arquivo(s) falharam no download")
        
        logger.info(f"\n{'='*70}\n")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description='Download automático de dados da lista piloto'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-baixar arquivos mesmo se já existirem'
    )
    parser.add_argument(
        '--only-pdb',
        action='store_true',
        help='Baixar apenas arquivos PDB'
    )
    parser.add_argument(
        '--only-bmrb',
        action='store_true',
        help='Baixar apenas arquivos BMRB'
    )
    parser.add_argument(
        '--pilot-list',
        default='data/pilot_list.yaml',
        help='Caminho para pilot_list.yaml (padrão: data/pilot_list.yaml)'
    )
    
    args = parser.parse_args()
    
    try:
        # Inicializar downloader
        downloader = PilotDataDownloader(args.pilot_list)
        
        # Executar downloads
        success, fail = downloader.download_all(
            force=args.force,
            only_pdb=args.only_pdb,
            only_bmrb=args.only_bmrb
        )
        
        # Imprimir sumário
        downloader.print_summary()
        
        # Exit code
        sys.exit(0 if fail == 0 else 1)
    
    except FileNotFoundError as e:
        logger.error(f"Erro: {e}")
        logger.error("Certifique-se de que data/pilot_list.yaml existe")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()