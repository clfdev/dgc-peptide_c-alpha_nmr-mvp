import json
import re
import urllib.request
from pathlib import Path

BASE = "https://api.bmrb.io/v2"
DB = "macromolecules"

def get_json(url: str):
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "bmrb-ca-lt60-script/1.0"
        }
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def clean_seq_and_len(seq: str) -> int:
    """
    BMRB sequences podem vir com quebras de linha, espaços, etc.
    Contamos apenas letras (A-Z). Ex.: remove números, espaços, '-' etc.
    """
    if not seq:
        return 0
    s = re.sub(r"[^A-Za-z]", "", seq)
    return len(s)

def main():
    # 1) Entradas que têm pelo menos um chemical shift com Atom_ID = "CA"
    # Endpoint: /search/get_id_by_tag_value/$tag_name/$tag_value?database=...
    url_ca = f"{BASE}/search/get_id_by_tag_value/Atom_chem_shift.Atom_ID/CA?database={DB}"
    ca_ids = get_json(url_ca)

    if not isinstance(ca_ids, list):
        raise RuntimeError(f"Resposta inesperada para CA IDs: {type(ca_ids)}")

    ca_set = set(str(x) for x in ca_ids)

    # 2) Sequências (one-letter) por entrada
    # Endpoint: /search/get_all_values_for_tag/$tag_name?database=...
    url_seqs = f"{BASE}/search/get_all_values_for_tag/Entity.Polymer_seq_one_letter_code?database={DB}"
    seq_map = get_json(url_seqs)

    if not isinstance(seq_map, dict):
        raise RuntimeError(f"Resposta inesperada para seq_map: {type(seq_map)}")

    # 3) Filtra entradas com alguma sequência < 60 aa
    lt60_set = set()
    for entry_id, seq_list in seq_map.items():
        # seq_list geralmente é lista de sequências (uma por entidade)
        if not isinstance(seq_list, list):
            continue

        lengths = [clean_seq_and_len(s) for s in seq_list]
        # considera apenas sequências com tamanho > 0
        lengths = [L for L in lengths if L > 0]

        if lengths and any(L < 60 for L in lengths):
            lt60_set.add(str(entry_id))

    # 4) Interseção: <60 aa E tem CA shift
    hit_ids = sorted(ca_set.intersection(lt60_set), key=lambda x: int(x) if x.isdigit() else x)

    print(f"Total BMRB entries (macromolecules) com comprimento < 60 e com Cα (Atom_ID=CA) shifts: {len(hit_ids)}")

    # salva lista
    out = Path("bmrb_peptides_lt60_with_CA_ids.txt")
    out.write_text("\n".join(hit_ids), encoding="utf-8")
    print(f"IDs salvos em: {out.resolve()}")

if __name__ == "__main__":
    main()
