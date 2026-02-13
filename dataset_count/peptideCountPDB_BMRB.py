import json
import time
import urllib.request
import urllib.error
from pathlib import Path

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"

def post_json(url: str, payload: dict, retries: int = 6, backoff: float = 2.5):
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "rcsb-peptides-lt60-bmrb/1.0"
    }
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            wait = backoff * attempt
            print(f"[post_json] erro: {e} -> retry em {wait:.1f}s ({attempt}/{retries})")
            time.sleep(wait)
    raise RuntimeError(f"Falha POST após {retries} tentativas: {url}")

def normalize_pdb(pdb_id: str) -> str:
    return str(pdb_id).strip().lower()

def fetch_all_pdb_ids(rows_per_page=1000):
    all_ids = []
    start = 0
    total = None
    page = 0

    while True:
        page += 1

        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_external_references.type",
                            "operator": "exact_match",
                            "value": "BMRB"
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "entity_poly.rcsb_entity_polymer_type",
                            "operator": "exact_match",
                            "value": "Protein"
                        }
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "entity_poly.rcsb_sample_sequence_length",
                            "operator": "range",
                            "value": {"from": 1, "to": 59}  # < 60
                        }
                    }
                ]
            },
            "return_type": "entry",
            "request_options": {"paginate": {"start": start, "rows": rows_per_page}}
        }

        resp = post_json(RCSB_SEARCH, query)

        if total is None:
            total = int(resp.get("total_count", 0))

        result_set = resp.get("result_set", [])
        batch = [normalize_pdb(r.get("identifier")) for r in result_set if r.get("identifier")]
        all_ids.extend(batch)

        print(f"[RCSB] page {page} | +{len(batch)} | acumulado {len(all_ids)} / total {total}")

        if not batch or len(all_ids) >= total:
            break

        start += rows_per_page

    # unique mantendo ordem
    seen = set()
    uniq = []
    for pid in all_ids:
        if pid and pid not in seen:
            seen.add(pid)
            uniq.append(pid)

    return uniq

def main():
    pdb_ids = fetch_all_pdb_ids(rows_per_page=1000)

    print("\n---- RESULTADO ----")
    print(f"Total PDB IDs (proteína <60 + link BMRB no RCSB): {len(pdb_ids)}")

    out_txt = Path("pdb_ids__peptides_lt60__has_bmrb_link.txt")
    out_csv = Path("pdb_ids__peptides_lt60__has_bmrb_link.csv")

    out_txt.write_text("\n".join(pdb_ids) + ("\n" if pdb_ids else ""), encoding="utf-8")
    out_csv.write_text("pdb_id\n" + "\n".join(pdb_ids) + ("\n" if pdb_ids else ""), encoding="utf-8")

    print("Arquivos gerados:")
    print(out_txt.resolve())
    print(out_csv.resolve())

if __name__ == "__main__":
    main()
