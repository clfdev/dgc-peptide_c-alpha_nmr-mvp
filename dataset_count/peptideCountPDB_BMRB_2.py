import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
import csv

INPUT_TXT = r"pdb_ids__peptides_lt60__has_bmrb_link.txt"

RCSB_CORE_ENTRY = "https://data.rcsb.org/rest/v1/core/entry/{}"

TIMEOUT_SEC = 120
RETRIES = 6
BACKOFF = 2.0
POLITE_DELAY = 0.03  # ajuste se precisar

OUT_CSV = "pdb_bmrb_from_rcsb__lt60.csv"
OUT_TXT = "pdb_bmrb_from_rcsb__lt60.txt"
OUT_UNRES = "pdb_bmrb_from_rcsb__unresolved.txt"

def normalize_pdb(line: str) -> str:
    s = line.strip().lower()
    if not s:
        return ""
    m = re.match(r"^([0-9][a-z0-9]{3})\b", s)
    return m.group(1) if m else ""

def get_json(url: str):
    headers = {
        "Accept": "application/json",
        "User-Agent": "rcsb-pdb-bmrb-extractor/1.0",
    }
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = f"HTTPError {e.code}: {e}"
            wait = BACKOFF * attempt
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = f"URLError/Timeout: {e}"
            wait = BACKOFF * attempt
            time.sleep(wait)
    raise RuntimeError(f"Falha após {RETRIES} tentativas: {url}\nÚltimo erro: {last_err}")

def extract_bmrb_ids_from_entry(entry_json: dict):
    """
    RCSB core entry geralmente traz rcsb_external_references como lista,
    com campos como: type, id, link, etc.
    Vamos capturar tudo que for type == 'BMRB' e pegar o 'id' (fallbacks inclusos).
    """
    bmrb_ids = set()

    refs = entry_json.get("rcsb_external_references", [])
    if isinstance(refs, list):
        for r in refs:
            if not isinstance(r, dict):
                continue
            if str(r.get("type", "")).strip().upper() == "BMRB":
                # chaves comuns
                cand = (
                    r.get("id")
                    or r.get("accession")
                    or r.get("accession_code")
                    or r.get("database_accession")
                )
                if cand is not None:
                    bmrb_ids.add(str(cand).strip())

    # fallback extra: alguns registros trazem BMRB em outros blocos,
    # mas normalmente rcsb_external_references resolve.
    return sorted(bmrb_ids, key=lambda x: int(x) if x.isdigit() else x)

def main():
    # 1) ler PDB IDs do arquivo
    pdb_ids = []
    with open(INPUT_TXT, "r", encoding="utf-8") as f:
        for line in f:
            pid = normalize_pdb(line)
            if pid:
                pdb_ids.append(pid)

    # unique preservando ordem
    seen = set()
    pdb_ids_uniq = []
    for pid in pdb_ids:
        if pid not in seen:
            seen.add(pid)
            pdb_ids_uniq.append(pid)

    print(f"PDB IDs lidos: {len(pdb_ids_uniq)}")

    # 2) abrir outputs
    out_csv = Path(OUT_CSV)
    out_txt = Path(OUT_TXT)
    out_unres = Path(OUT_UNRES)

    unresolved = []

    rows = []
    for i, pdb_id in enumerate(pdb_ids_uniq, start=1):
        url = RCSB_CORE_ENTRY.format(pdb_id)
        try:
            entry = get_json(url)
            bmrb_ids = extract_bmrb_ids_from_entry(entry)
        except Exception as e:
            bmrb_ids = []
            unresolved.append((pdb_id, str(e).replace("\n", " ")))

        rows.append((pdb_id, bmrb_ids))

        if i % 200 == 0:
            print(f"Processados {i}/{len(pdb_ids_uniq)}")

        time.sleep(POLITE_DELAY)

    # 3) salvar CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pdb_id", "bmrb_ids"])
        for pdb_id, bmrb_ids in rows:
            w.writerow([pdb_id, ";".join(bmrb_ids)])

    # 4) salvar TXT “lado a lado” (bom pra triagem manual)
    # formato: pdb_id <tab> bmrb_ids
    lines = []
    for pdb_id, bmrb_ids in rows:
        lines.append(f"{pdb_id}\t{';'.join(bmrb_ids)}")
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 5) salvar unresolved
    if unresolved:
        out_unres.write_text(
            "\n".join([f"{p}\t{err}" for p, err in unresolved]) + "\n",
            encoding="utf-8"
        )
    else:
        out_unres.write_text("", encoding="utf-8")

    n_with = sum(1 for _, b in rows if b)
    n_without = sum(1 for _, b in rows if not b)

    print("---- Final ----")
    print(f"Com BMRB extraído no RCSB: {n_with}")
    print(f"Sem BMRB extraído (inesperado): {n_without}")
    print(f"Falhas/timeout (reprocessar): {len(unresolved)}")
    print("Arquivos gerados:")
    print(out_csv.resolve())
    print(out_txt.resolve())
    print(out_unres.resolve())

if __name__ == "__main__":
    main()
