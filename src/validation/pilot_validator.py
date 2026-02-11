# src/validation/pilot_validator.py
"""
Automatic validator for PDB-BMRB pairs (MVP phase 0).
Checks availability, basic PDB quality, and basic BMRB accessibility.

ASCII-only version (no accents, no unicode symbols) to avoid Windows cp1252 errors.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import requests

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    PASS = "[PASS]"
    FAIL = "[FAIL]"
    WARNING = "[WARN]"
    SKIP = "[SKIP]"


@dataclass
class ValidationResult:
    status: ValidationStatus
    message: str
    value: Optional[any] = None


@dataclass
class PilotEntryReport:
    pdb_id: str
    bmrb_id: str
    checks: Dict[str, ValidationResult]
    is_valid: bool
    overall_message: str

    def __str__(self) -> str:
        lines = [
            f"\n{'='*70}",
            f"VALIDACAO: {self.pdb_id} <-> BMRB {self.bmrb_id}",
            f"{'='*70}",
            f"Status Geral: {'APROVADO' if self.is_valid else 'REJEITADO'}",
            "\nChecks Individuais:",
        ]

        for check_name, result in self.checks.items():
            lines.append(f"  {result.status.value} {check_name}: {result.message}")
            if result.value is not None:
                lines.append(f"      Valor: {result.value}")

        lines.append(f"\n{self.overall_message}")
        lines.append(f"{'='*70}\n")
        return "\n".join(lines)


class PilotValidator:
    """
    Main validator.

    Important:
    - Does NOT depend on webapi.bmrb.io (DNS issues on some networks).
    - BMRB checks are done via FTP HTTP NMR-STAR .str download URLs.
    """

    # PDB metadata via GraphQL
    RCSB_API = "https://data.rcsb.org/graphql"
    RCSB_PDB_URL = "https://files.rcsb.org/download/{}.pdb"

    # BMRB NMR-STAR (.str) URLs (FTP over HTTP)
    BMRB_STAR_URL_A = "https://bmrb.io/ftp/pub/bmrb/entry_lists/nmr-star3.1/bmr{}.str"
    BMRB_STAR_URL_B = "https://bmrb.io/ftp/pub/bmrb/entry_directories/bmr{0}/bmr{0}_3.str"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "DGC-NMR-Validator/0.1"})

    def validate_pilot_entry(
        self,
        pdb_id: str,
        bmrb_id: str,
        min_length: int = 9,
        max_length: int = 60,
        max_resolution: float = 2.5,
        min_ca_coverage: float = 0.70,
    ) -> PilotEntryReport:
        pdb_id = pdb_id.upper()
        bmrb_id = str(bmrb_id).strip()
        checks: Dict[str, ValidationResult] = {}

        # 1) PDB availability
        logger.info(f"Validando PDB {pdb_id}...")
        checks["1_pdb_exists"] = self._check_pdb_availability(pdb_id)

        # 2) BMRB availability (via FTP HTTP .str)
        logger.info(f"Validando BMRB {bmrb_id}...")
        checks["2_bmrb_exists"] = self._check_bmrb_availability(bmrb_id)

        # If either is missing, stop early
        if (
            checks["1_pdb_exists"].status == ValidationStatus.FAIL
            or checks["2_bmrb_exists"].status == ValidationStatus.FAIL
        ):
            return PilotEntryReport(
                pdb_id=pdb_id,
                bmrb_id=bmrb_id,
                checks=checks,
                is_valid=False,
                overall_message="Entrada ou BMRB nao encontrados",
            )

        # 3) PDB metadata
        pdb_metadata = self._get_pdb_metadata(pdb_id)

        # 4) Resolution
        checks["3_resolution"] = self._check_resolution(pdb_metadata, max_resolution)

        # 5) Experimental method
        checks["4_method"] = self._check_experimental_method(pdb_metadata)

        # 6) Protein chain count
        checks["5_chain_count"] = self._check_chain_count(pdb_metadata)

        # 7) Length
        checks["6_length"] = self._check_length(pdb_metadata, min_length, max_length)

        # 8) BMRB data (download .str text) - no webapi
        bmrb_metadata = self._get_bmrb_metadata(bmrb_id)

        # 9) CA coverage (still stub for MVP -> WARN)
        checks["7_ca_coverage"] = self._check_ca_coverage(bmrb_metadata, min_ca_coverage)

        # 10) Sequence match (still stub for MVP -> WARN until STAR parsing is implemented)
        checks["8_sequence_match"] = self._check_sequence_correspondence(pdb_metadata, bmrb_metadata)

        # 11) Structural quality (R-free if available)
        checks["9_quality"] = self._check_structural_quality(pdb_metadata)

        # Critical checks
        critical_checks = [
            "1_pdb_exists",
            "2_bmrb_exists",
            "3_resolution",
            "5_chain_count",
            "6_length",
            "7_ca_coverage",
            "8_sequence_match",
        ]

        is_valid = all(
            checks[c].status in (ValidationStatus.PASS, ValidationStatus.WARNING)
            for c in critical_checks
            if c in checks
        )

        if is_valid:
            overall_message = f"Entrada {pdb_id} <-> {bmrb_id} APROVADA para lista piloto"
        else:
            failed = [k for k, v in checks.items() if v.status == ValidationStatus.FAIL]
            overall_message = f"Entrada REJEITADA. Falhas: {', '.join(failed)}"

        return PilotEntryReport(
            pdb_id=pdb_id,
            bmrb_id=bmrb_id,
            checks=checks,
            is_valid=is_valid,
            overall_message=overall_message,
        )

    # -------------------- Checks --------------------

    def _check_pdb_availability(self, pdb_id: str) -> ValidationResult:
        try:
            url = self.RCSB_PDB_URL.format(pdb_id)
            response = self.session.head(url, timeout=10)
            if response.status_code == 200:
                return ValidationResult(
                    status=ValidationStatus.PASS,
                    message=f"PDB {pdb_id} disponivel para download",
                    value=url,
                )
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message=f"PDB {pdb_id} nao encontrado (HTTP {response.status_code})",
            )
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message=f"Erro ao acessar PDB: {str(e)}",
            )

    def _check_bmrb_availability(self, bmrb_id: str) -> ValidationResult:
        """
        Check if BMRB entry exists by trying to reach the NMR-STAR .str file
        via the official FTP HTTP endpoints.
        """
        urls_to_try = [
            self.BMRB_STAR_URL_A.format(bmrb_id),
            self.BMRB_STAR_URL_B.format(bmrb_id),
        ]

        last_error = None

        for url in urls_to_try:
            try:
                # Prefer HEAD (fast). Some servers may block HEAD.
                resp = self.session.head(url, timeout=10, allow_redirects=True)

                if resp.status_code == 200:
                    return ValidationResult(
                        status=ValidationStatus.PASS,
                        message=f"BMRB {bmrb_id} disponivel (NMR-STAR .str)",
                        value=url,
                    )

                # If HEAD blocked, try small GET
                if resp.status_code in (403, 405):
                    resp2 = self.session.get(url, timeout=15, stream=True, allow_redirects=True)
                    if resp2.status_code == 200:
                        resp2.close()
                        return ValidationResult(
                            status=ValidationStatus.PASS,
                            message=f"BMRB {bmrb_id} disponivel (NMR-STAR .str)",
                            value=url,
                        )
                    last_error = f"HTTP {resp2.status_code}"
                    resp2.close()
                else:
                    last_error = f"HTTP {resp.status_code}"

            except Exception as e:
                last_error = str(e)

        return ValidationResult(
            status=ValidationStatus.FAIL,
            message=f"Erro ao acessar BMRB {bmrb_id} via FTP HTTP ({last_error})",
            value=urls_to_try,
        )

    def _get_pdb_metadata(self, pdb_id: str) -> Dict:
        query = f"""
        {{
          entry(entry_id: "{pdb_id}") {{
            struct {{
              title
            }}
            rcsb_entry_info {{
              resolution_combined
              experimental_method
              polymer_entity_count_protein
            }}
            polymer_entities {{
              entity_poly {{
                pdbx_seq_one_letter_code_can
              }}
              rcsb_polymer_entity {{
                pdbx_number_of_molecules
              }}
            }}
            exptl {{
              method
            }}
            refine {{
              ls_R_factor_R_free
              ls_R_factor_R_work
            }}
          }}
        }}
        """
        try:
            response = self.session.post(self.RCSB_API, json={"query": query}, timeout=15)
            if response.status_code == 200:
                data = response.json()
                return data.get("data", {}).get("entry", {}) or {}
            logger.warning(f"Falha ao obter metadados PDB: {response.status_code}")
            return {}
        except Exception as e:
            logger.error(f"Erro ao buscar metadados PDB: {e}")
            return {}

    def _get_bmrb_metadata(self, bmrb_id: str) -> Dict:
        """
        MVP: downloads the STAR file text and returns minimal metadata.
        No dependency on webapi.bmrb.io.
        """
        bmrb_id = str(bmrb_id).strip()
        urls_to_try = [
            self.BMRB_STAR_URL_A.format(bmrb_id),
            self.BMRB_STAR_URL_B.format(bmrb_id),
        ]

        last_error = None
        for url in urls_to_try:
            try:
                resp = self.session.get(url, timeout=20)
                if resp.status_code == 200 and resp.text:
                    return {"bmrb_id": bmrb_id, "source_url": url, "star_text": resp.text}
                last_error = f"HTTP {resp.status_code}"
            except Exception as e:
                last_error = str(e)

        logger.error(f"Erro ao buscar metadados BMRB via FTP HTTP: {last_error}")
        return {}

    def _check_resolution(self, pdb_metadata: Dict, max_resolution: float) -> ValidationResult:
        try:
            resolution = pdb_metadata.get("rcsb_entry_info", {}).get("resolution_combined", [None])
            if isinstance(resolution, list):
                resolution = resolution[0] if resolution else None

            if resolution is None:
                return ValidationResult(
                    status=ValidationStatus.WARNING,
                    message="Resolucao nao disponivel (estrutura NMR?)",
                )

            resolution = float(resolution)
            if resolution <= max_resolution:
                return ValidationResult(
                    status=ValidationStatus.PASS,
                    message=f"Resolucao excelente: {resolution:.2f} A",
                    value=resolution,
                )
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message=f"Resolucao {resolution:.2f} A > limite {max_resolution} A",
                value=resolution,
            )
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message=f"Erro ao verificar resolucao: {str(e)}",
            )

    def _check_experimental_method(self, pdb_metadata: Dict) -> ValidationResult:
        try:
            methods = pdb_metadata.get("rcsb_entry_info", {}).get("experimental_method", [])
            if not methods:
                methods = [m.get("method") for m in pdb_metadata.get("exptl", [])]

            methods_str = ", ".join(methods) if methods else "Desconhecido"
            if any(("X-RAY" in m.upper() or "NMR" in m.upper()) for m in methods):
                return ValidationResult(
                    status=ValidationStatus.PASS,
                    message=f"Metodo aceito: {methods_str}",
                    value=methods,
                )
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message=f"Metodo incomum: {methods_str}",
                value=methods,
            )
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message=f"Erro ao verificar metodo: {str(e)}",
            )

    def _check_chain_count(self, pdb_metadata: Dict) -> ValidationResult:
        try:
            protein_count = pdb_metadata.get("rcsb_entry_info", {}).get("polymer_entity_count_protein", 0)

            if protein_count == 1:
                return ValidationResult(
                    status=ValidationStatus.PASS,
                    message="Estrutura com 1 cadeia proteica",
                    value=protein_count,
                )
            if protein_count > 1:
                return ValidationResult(
                    status=ValidationStatus.WARNING,
                    message=f"Multiplas cadeias ({protein_count}). Usar cadeia A?",
                    value=protein_count,
                )
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="Nenhuma cadeia proteica encontrada",
                value=protein_count,
            )
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message=f"Erro ao verificar cadeias: {str(e)}",
            )

    def _check_length(self, pdb_metadata: Dict, min_length: int, max_length: int) -> ValidationResult:
        try:
            entities = pdb_metadata.get("polymer_entities", [])
            if not entities:
                return ValidationResult(
                    status=ValidationStatus.FAIL,
                    message="Nenhuma entidade polimerica encontrada",
                )

            sequence = entities[0].get("entity_poly", {}).get("pdbx_seq_one_letter_code_can", "")
            sequence = re.sub(r"\s+", "", sequence)
            length = len(sequence)

            if min_length <= length <= max_length:
                return ValidationResult(
                    status=ValidationStatus.PASS,
                    message=f"Comprimento adequado: {length} residuos",
                    value=length,
                )
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message=f"Comprimento {length} fora do range [{min_length}, {max_length}]",
                value=length,
            )
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message=f"Erro ao verificar comprimento: {str(e)}",
            )

    def _check_ca_coverage(self, bmrb_metadata: Dict, min_coverage: float) -> ValidationResult:
        """
        MVP stub: real coverage requires parsing STAR text loop _Atom_chem_shift.
        For now, WARN, but we keep the pipeline running.
        """
        _ = min_coverage
        if not bmrb_metadata or "star_text" not in bmrb_metadata:
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message="BMRB STAR nao disponivel para checar cobertura CA (download falhou?)",
            )
        return ValidationResult(
            status=ValidationStatus.WARNING,
            message="Cobertura CA precisa verificacao via parsing do arquivo .str (MVP stub)",
        )

    def _check_sequence_correspondence(self, pdb_metadata: Dict, bmrb_metadata: Dict) -> ValidationResult:
        """
        MVP stub: proper sequence match requires parsing STAR entity sequence and alignment.
        For now, WARN.
        """
        if not pdb_metadata:
            return ValidationResult(status=ValidationStatus.WARNING, message="Metadados PDB ausentes")

        if not bmrb_metadata or "star_text" not in bmrb_metadata:
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message="Sequencia BMRB precisa verificacao via parsing do arquivo .str (download ausente)",
            )

        return ValidationResult(
            status=ValidationStatus.WARNING,
            message="Correspondencia de sequencias PDB-BMRB requer parsing STAR + alinhamento (MVP stub)",
        )

    def _check_structural_quality(self, pdb_metadata: Dict) -> ValidationResult:
        try:
            refine = pdb_metadata.get("refine", [])
            if not refine:
                return ValidationResult(
                    status=ValidationStatus.WARNING,
                    message="Metricas de refinamento nao disponiveis (estrutura NMR?)",
                )

            r_free = refine[0].get("ls_R_factor_R_free")
            r_work = refine[0].get("ls_R_factor_R_work")

            if r_free is None:
                return ValidationResult(status=ValidationStatus.WARNING, message="R-free nao disponivel")

            r_free = float(r_free)
            if r_free < 0.25:
                return ValidationResult(
                    status=ValidationStatus.PASS,
                    message=f"Excelente qualidade: R-free = {r_free:.3f}",
                    value={"r_free": r_free, "r_work": r_work},
                )
            if r_free < 0.30:
                return ValidationResult(
                    status=ValidationStatus.WARNING,
                    message=f"Qualidade aceitavel: R-free = {r_free:.3f}",
                    value={"r_free": r_free, "r_work": r_work},
                )
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message=f"Qualidade baixa: R-free = {r_free:.3f}",
                value={"r_free": r_free, "r_work": r_work},
            )
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message=f"Erro ao verificar qualidade: {str(e)}",
            )


def validate_pilot_list(pilot_list: List[Tuple[str, str]]) -> None:
    validator = PilotValidator()
    results: List[PilotEntryReport] = []

    print("\n" + "=" * 70)
    print("VALIDACAO DA LISTA PILOTO")
    print("=" * 70)

    for pdb_id, bmrb_id in pilot_list:
        report = validator.validate_pilot_entry(pdb_id, bmrb_id)
        results.append(report)
        print(report)

    approved = sum(1 for r in results if r.is_valid)
    total = len(results)

    print("\n" + "=" * 70)
    print(f"SUMARIO FINAL: {approved}/{total} entradas aprovadas")
    print("=" * 70)

    print("\n[APPROVED]:")
    for r in results:
        if r.is_valid:
            print(f"  - {r.pdb_id} <-> BMRB {r.bmrb_id}")

    rejected = [r for r in results if not r.is_valid]
    if rejected:
        print("\n[REJECTED]:")
        for r in rejected:
            print(f"  - {r.pdb_id} <-> BMRB {r.bmrb_id}")
            print(f"    Motivo: {r.overall_message}")


if __name__ == "__main__":
    PILOT_LIST = [
        ("1VII", "5713"),  # Villin headpiece
        ("1LE1", "5387"),  # Trpzip-2
        ("1E0L", "4737"),  # WW domain
        ("1UBQ", "6457"),  # Ubiquitin (use residues 1-30 later in parsing stage)
        ("2MAG", "6031"),  # Magainin-2
    ]

    validate_pilot_list(PILOT_LIST)
