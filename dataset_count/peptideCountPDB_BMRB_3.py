import csv
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

# ====== ENTRADA (seu arquivo) ======
INPUT_TXT = r"pdb_bmrb_from_rcsb__lt60.txt"  # <- use este

# ====== BMRB API ======
BASE_BMRB = "https://api.bmrb.io/v2"
DB = "macromolecules"

# ====== Rede / robustez ======
TIMEOUT_SEC = 180
RETRIES = 6
BACKOFF = 2.0

# ====== Saídas ======
OUT_YES_TXT = "pdb_bmrb__lt60__CA_yes.txt"
OUT_NO_TXT  = "pdb_bmrb__lt60__CA_no.txt"
OUT_ALL_CSV = "pdb_bmrb__lt60__CA_summary.csv"

def get_json(url: str):
    headers = {
        "Accept": "application/json",
        "User-Agent": "bmrb-ca-filter-from-pairs/1.0"
    }
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = f"HTTPError {e.code}: {e}"
            # 403 = rate limit -> espera mais
            wait = 12.0 if e.code == 403 else BACKOFF * attempt
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = f"URLError/Timeout: {e}"
            wait = BACKOFF * attempt
            time.sleep(wait)

    raise RuntimeError(f"Falha após {RETRIES} tentativas: {url}\nÚltimo erro: {last_err}")

def parse_line(line: str):
    """
    Espera:
      pdb_id<TAB>bmrb1;bmrb2;...
    """
    line = line.strip()
    if not line:
        return None, []
    parts = line.split("\t")
    pdb = parts[0].strip().lower()
    bmrb_str = parts[1].strip() if len(parts) > 1 else ""
    bmrb_ids = [x.strip() for x in bmrb_str.split(";") if x.strip()]
    return pdb, bmrb_ids

def main():
    # 1) Baixar 1 vez o conjunto de BMRB IDs que têm CA
    url_ca = f"{BASE_BMRB}/search/get_id_by_tag_value/Atom_chem_shift.Atom_ID/CA?database={DB}"
    ca_ids = get_json(url_ca)

    if not isinstance(ca_ids, list):
        raise RuntimeError(f"Resposta inesperada para CA IDs: {type(ca_ids)}")

    ca_set = set(str(x).strip() for x in ca_ids)
    print(f"BMRB IDs com Atom_ID=CA: {len(ca_set)}")

    # 2) Ler pares PDB↔BMRB do seu TXT
    rows = []
    yes_lines = []
    no_lines = []

    with open(INPUT_TXT, "r", encoding="utf-8") as f:
        for line in f:
            pdb, bmrb_ids = parse_line(line)
            if not pdb:
                continue

            # marca se algum BMRB do PDB está no conjunto CA
            has_ca = any(b in ca_set for b in bmrb_ids)

            rows.append({
                "pdb_id": pdb,
                "bmrb_ids": ";".join(bmrb_ids),
                "has_CA": "YES" if has_ca else "NO"
            })

            if has_ca:
                yes_lines.append(f"{pdb}\t{';'.join(bmrb_ids)}")
            else:
                no_lines.append(f"{pdb}\t{';'.join(bmrb_ids)}")

    # 3) Salvar saídas
    Path(OUT_YES_TXT).write_text("\n".join(yes_lines) + ("\n" if yes_lines else ""), encoding="utf-8")
    Path(OUT_NO_TXT).write_text("\n".join(no_lines) + ("\n" if no_lines else ""), encoding="utf-8")

    with open(OUT_ALL_CSV, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=["pdb_id", "bmrb_ids", "has_CA"])
        w.writeheader()
        w.writerows(rows)

    print("---- Final ----")
    print(f"Total pares PDB lidos: {len(rows)}")
    print(f"Tem CA (YES): {len(yes_lines)}")
    print(f"Não tem CA (NO): {len(no_lines)}")
    print("Arquivos gerados:")
    print(Path(OUT_YES_TXT).resolve())
    print(Path(OUT_NO_TXT).resolve())
    print(Path(OUT_ALL_CSV).resolve())

if __name__ == "__main__":
    main()
