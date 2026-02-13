import json
import urllib.request

query = {
  "query": {
    "type": "group",
    "logical_operator": "and",
    "nodes": [
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
          "value": {"from": 1, "to": 59}   # < 60
        }
      }
    ]
  },
  "return_type": "polymer_entity",
  "request_options": {"return_counts": True}
}

data = json.dumps(query).encode("utf-8")

req = urllib.request.Request(
    "https://search.rcsb.org/rcsbsearch/v2/query",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urllib.request.urlopen(req) as resp:
    out = json.loads(resp.read().decode("utf-8"))

print(out.get("total_count"))
