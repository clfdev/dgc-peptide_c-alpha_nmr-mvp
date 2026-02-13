# scripts/evaluate_predictions.py

import argparse
import json
import math
import os
from pathlib import Path
import shlex
from collections import defaultdict

import pandas as pd

'''
Exemplo de uso:
python scripts/evaluate_predictions.py --pred_csv tests/1L2Y_shifts_predicted_v2.csv --bmrb_str tests/1L2Y_bmr5293.str --out_dir results/eval_1L2Y
'''

def parse_bmrb_ca13c_shifts(
    bmrb_str_path: str,
    debug: bool = False,
    debug_n: int = 20,
):
    """
    Parseia um arquivo BMRB NMR-STAR (.str) e extrai chemical shifts de 13C(CA)
    do loop _Atom_chem_shift.

    Retorna:
        shifts_by_seqid: dict[int -> float]  # media se duplicado
        debug_info: dict com contagens (atom_id, atom_type, iso) e flags
    """
    path = Path(bmrb_str_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo BMRB .str nao encontrado: {bmrb_str_path}")

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    shifts_acc = defaultdict(list)

    seen_atom_id = defaultdict(int)
    seen_atom_type = defaultdict(int)
    seen_iso = defaultdict(int)

    found_target_loop = False
    debug_printed = 0

    i = 0
    n = len(lines)

    def is_tag_line(s: str) -> bool:
        s = s.strip()
        return s.startswith("_")

    def is_data_line(s: str) -> bool:
        s = s.strip()
        if not s:
            return False
        if s.startswith("_"):
            return False
        if s.startswith("#"):
            return False
        if s.lower().startswith("loop_"):
            return False
        if s.lower().startswith("stop_"):
            return False
        if s.lower().startswith("save_"):
            return False
        return True

    while i < n:
        s = lines[i].strip()

        if s.lower().startswith("loop_"):
            # Coleta tags do loop
            i += 1
            tags = []
            while i < n and is_tag_line(lines[i]):
                tags.append(lines[i].strip())
                i += 1

            # Verifica se esse loop parece ser o _Atom_chem_shift
            # Precisamos pelo menos dessas tags:
            # _Atom_chem_shift.Atom_ID
            # _Atom_chem_shift.Atom_type
            # _Atom_chem_shift.Val
            # e algum ID sequencial: Seq_ID ou Comp_index_ID
            tag_set = set(tags)

            def find_tag(*candidates):
                for c in candidates:
                    if c in tag_set:
                        return c
                return None

            atom_id_tag = find_tag("_Atom_chem_shift.Atom_ID")
            atom_type_tag = find_tag("_Atom_chem_shift.Atom_type")
            val_tag = find_tag("_Atom_chem_shift.Val")
            seq_tag = find_tag("_Atom_chem_shift.Seq_ID", "_Atom_chem_shift.Comp_index_ID")
            iso_tag = find_tag("_Atom_chem_shift.Atom_isotope_number")

            # Se não é o loop alvo, pula leitura de dados até stop_
            if not (atom_id_tag and atom_type_tag and val_tag and seq_tag):
                # consumir linhas de dados até stop_ ou novo loop/save_
                while i < n and not lines[i].strip().lower().startswith(("stop_", "loop_", "save_")):
                    i += 1
                # se parou em stop_, consome e segue
                if i < n and lines[i].strip().lower().startswith("stop_"):
                    i += 1
                continue

            # Este é o loop alvo
            found_target_loop = True

            col = {tag: idx for idx, tag in enumerate(tags)}
            # Agora parseia dados até stop_
            while i < n:
                raw = lines[i].strip()
                if raw.lower().startswith("stop_"):
                    i += 1
                    break
                if raw.lower().startswith(("loop_", "save_")):
                    # loop sem stop_ (raro) -> volta para while externo
                    break
                if not is_data_line(raw):
                    i += 1
                    continue

                # Tokenização robusta com shlex (respeita aspas)
                try:
                    parts = shlex.split(raw, posix=True)
                except Exception:
                    parts = raw.split()

                # Se a linha não tem colunas suficientes, pode ser continuação multiline (MVP: ignora)
                if len(parts) < len(tags):
                    i += 1
                    continue

                # Extrai campos
                atom_id = parts[col[atom_id_tag]].strip("\"'").upper()
                atom_type = parts[col[atom_type_tag]].strip("\"'").upper()

                seen_atom_id[atom_id] += 1
                seen_atom_type[atom_type] += 1

                # isótopo (se existir)
                iso = None
                if iso_tag is not None:
                    iso_raw = parts[col[iso_tag]].strip("\"'")
                    # pode vir "." ou "?"
                    if iso_raw.isdigit():
                        iso = int(iso_raw)
                        seen_iso[str(iso)] += 1
                    else:
                        seen_iso[iso_raw] += 1

                # Debug: imprime primeiras N linhas do loop alvo
                if debug and debug_printed < debug_n:
                    seq_raw = parts[col[seq_tag]].strip("\"'")
                    val_raw = parts[col[val_tag]].strip("\"'")
                    print("DEBUG(row):", "atom_id=", atom_id, "atom_type=", atom_type, "iso=", iso, "seq=", seq_raw, "val=", val_raw)
                    debug_printed += 1

                # Filtros: CA + C (+ 13 se tiver iso)
                if atom_id != "CA":
                    i += 1
                    continue
                if atom_type != "C":
                    i += 1
                    continue
                if iso is not None and iso != 13:
                    i += 1
                    continue

                # seq id
                seq_raw = parts[col[seq_tag]].strip("\"'")
                if not seq_raw or seq_raw in (".", "?"):
                    i += 1
                    continue

                try:
                    seq_id = int(seq_raw)
                except ValueError:
                    i += 1
                    continue

                # valor shift
                val_raw = parts[col[val_tag]].strip("\"'")
                if not val_raw or val_raw in (".", "?"):
                    i += 1
                    continue

                try:
                    shift_val = float(val_raw)
                except ValueError:
                    i += 1
                    continue

                shifts_acc[seq_id].append(shift_val)

                i += 1

            # terminou loop alvo (ou encontrou loop_/save_ sem stop_)
            # (não damos break porque pode haver mais loops, mas para shifts CA basta o primeiro)
            continue

        i += 1

    if not found_target_loop:
        raise RuntimeError(
            "Nao encontrei loop _Atom_chem_shift com as tags necessarias "
            "(Atom_ID, Atom_type, Val e Seq_ID/Comp_index_ID)."
        )

    # Média para duplicados
    shifts_by_seqid = {}
    for seq_id, vals in shifts_acc.items():
        shifts_by_seqid[seq_id] = sum(vals) / len(vals)

    debug_info = {
        "found_target_loop": found_target_loop,
        "unique_seq_ids": len(shifts_by_seqid),
        "seen_atom_id_top": dict(sorted(seen_atom_id.items(), key=lambda x: -x[1])[:20]),
        "seen_atom_type_top": dict(sorted(seen_atom_type.items(), key=lambda x: -x[1])[:20]),
        "seen_iso_top": dict(sorted(seen_iso.items(), key=lambda x: -x[1])[:20]),
    }

    if debug:
        print("DEBUG(atom_id counts top):", debug_info["seen_atom_id_top"])
        print("DEBUG(atom_type counts top):", debug_info["seen_atom_type_top"])
        print("DEBUG(isotope counts top):", debug_info["seen_iso_top"])
        print("DEBUG(unique CA-13C seq_ids):", debug_info["unique_seq_ids"])

    return shifts_by_seqid, debug_info


def compute_metrics(y_true, y_pred):
    y_true = list(y_true)
    y_pred = list(y_pred)

    n = len(y_true)
    if n == 0:
        return {
            "n": 0,
            "MAE": None,
            "RMSE": None,
            "MSE": None,
            "MRE_percent": None,
        }

    abs_err = [abs(a - b) for a, b in zip(y_true, y_pred)]
    mae = sum(abs_err) / n
    mse = sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n
    rmse = math.sqrt(mse)

    # MRE%: média do erro relativo percentual; ignora y_true ~ 0
    rels = []
    for a, b in zip(y_true, y_pred):
        if abs(a) < 1e-9:
            continue
        rels.append(abs(a - b) / abs(a))
    mre = (sum(rels) / len(rels) * 100.0) if rels else None

    return {
        "n": n,
        "MAE": mae,
        "RMSE": rmse,
        "MSE": mse,
        "MRE_percent": mre,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compara shifts preditos (CSV) com shifts experimentais BMRB (.str) para CA-13C e calcula metricas."
    )
    parser.add_argument("--pred_csv", required=True, help="CSV com shifts preditos (saida do seu predict_shifts).")
    parser.add_argument("--bmrb_str", required=True, help="Arquivo BMRB NMR-STAR .str.")
    parser.add_argument("--out_dir", required=True, help="Diretorio de saida (sera criado se nao existir).")
    parser.add_argument("--debug", action="store_true", help="Imprime debug do parsing do BMRB.")
    parser.add_argument("--debug_n", type=int, default=20, help="Numero de linhas de debug (padrao 20).")
    args = parser.parse_args()

    pred_csv = Path(args.pred_csv)
    bmrb_str = Path(args.bmrb_str)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_csv.exists():
        raise FileNotFoundError(f"pred_csv nao encontrado: {pred_csv}")
    if not bmrb_str.exists():
        raise FileNotFoundError(f"bmrb_str nao encontrado: {bmrb_str}")

    # 1) Ler preditos
    df_pred = pd.read_csv(pred_csv)

    # Checagem de colunas esperadas
    required_pred_cols = {"residue_number", "shift_ca_predicted"}
    missing = required_pred_cols - set(df_pred.columns)
    if missing:
        raise ValueError(
            f"CSV de predicao nao tem colunas obrigatorias {sorted(required_pred_cols)}. "
            f"Faltando: {sorted(missing)}. Colunas encontradas: {df_pred.columns.tolist()}"
        )

    df_pred = df_pred.copy()
    df_pred["residue_number"] = pd.to_numeric(df_pred["residue_number"], errors="coerce").astype("Int64")
    df_pred = df_pred.dropna(subset=["residue_number", "shift_ca_predicted"])

    # 2) Ler BMRB (CA-13C)
    shifts_by_seqid, debug_info = parse_bmrb_ca13c_shifts(
        str(bmrb_str), debug=args.debug, debug_n=args.debug_n
    )

    # 3) Montar df experimental
    df_exp = pd.DataFrame(
        {"residue_number": list(shifts_by_seqid.keys()), "shift_ca_exp": list(shifts_by_seqid.values())}
    )
    df_exp["residue_number"] = pd.to_numeric(df_exp["residue_number"], errors="coerce").astype("Int64")
    df_exp = df_exp.dropna(subset=["residue_number", "shift_ca_exp"])

    # 4) Merge
    merged = df_pred.merge(df_exp, on="residue_number", how="inner")

    print(f"bmrb_str: {bmrb_str}")
    print(f"merged rows (CA-13C): {len(merged)}")

    # 5) Métricas
    metrics = compute_metrics(merged["shift_ca_exp"].tolist(), merged["shift_ca_predicted"].tolist())

    print(f"MAE  : {metrics['MAE']:.4f} ppm" if metrics["MAE"] is not None else "MAE  : N/A")
    print(f"RMSE : {metrics['RMSE']:.4f} ppm" if metrics["RMSE"] is not None else "RMSE : N/A")
    print(f"MSE  : {metrics['MSE']:.4f} (ppm^2)" if metrics["MSE"] is not None else "MSE  : N/A")
    if metrics["MRE_percent"] is not None:
        print(f"MRE% : {metrics['MRE_percent']:.4f} %")
    else:
        print("MRE% : N/A")

    # 6) Salvar saídas
    merged_out = out_dir / "merged.csv"
    merged.to_csv(merged_out, index=False)

    metrics_out_json = out_dir / "metrics.json"
    metrics_payload = {
        "pred_csv": str(pred_csv),
        "bmrb_str": str(bmrb_str),
        "merged_rows": int(len(merged)),
        "metrics": metrics,
        "bmrb_debug": debug_info,
    }
    metrics_out_json.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    metrics_out_txt = out_dir / "metrics.txt"
    lines = [
        f"pred_csv: {pred_csv}",
        f"bmrb_str: {bmrb_str}",
        f"merged rows (CA-13C): {len(merged)}",
        "",
        f"MAE  : {metrics['MAE']}" if metrics["MAE"] is not None else "MAE  : N/A",
        f"RMSE : {metrics['RMSE']}" if metrics["RMSE"] is not None else "RMSE : N/A",
        f"MSE  : {metrics['MSE']}" if metrics["MSE"] is not None else "MSE  : N/A",
        f"MRE% : {metrics['MRE_percent']}" if metrics["MRE_percent"] is not None else "MRE% : N/A",
        "",
        "BMRB debug (tops):",
        json.dumps(debug_info, indent=2),
    ]
    metrics_out_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {merged_out}")
    print(f"Saved: {metrics_out_json}")
    print(f"Saved: {metrics_out_txt}")


if __name__ == "__main__":
    main()
