# scripts/train_model.py

"""
Treina modelo Ridge Regression para predição de chemical shifts CA.

Estratégia:
- GroupKFold por pdb_id (leave-one-structure-out)
- Normalização de features (StandardScaler)
- Ridge com CV interna para escolha de lambda
- Métricas: MAE, RMSE, R² (global e por estrutura)

Input: data/processed/pilot_dataset_features.csv
Output:
  - results/predictions.csv
  - results/model_performance.csv
  - results/summary.txt
  - models/ridge_model.pkl
  - models/scaler.pkl
  - models/metadata.json
"""

import sys
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_and_evaluate():
    """Treina modelo Ridge com GroupKFold e avalia resultados."""

    print(f"\n{'='*70}")
    print("TREINAMENTO DO MODELO RIDGE - MVP FASE 0")
    print(f"{'='*70}\n")

    # Carregar dataset
    input_path = Path("data/processed/pilot_dataset_features.csv")

    if not input_path.exists():
        print(f"Erro: {input_path} não encontrado!")
        print("Execute calculate_features.py primeiro.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"Dataset carregado: {len(df)} amostras, {df['pdb_id'].nunique()} estruturas")

    # Definir features e target
    feature_cols = [
        "dist_to_center",
        "dist_to_nearest_neighbor",
        "local_density",
        "dist_to_n_term",
        "dist_to_c_term",
    ]
    target_col = "shift_ca_experimental"

    # Matriz X / vetor y / grupos
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df["pdb_id"].values

    print(f"\nFeatures utilizadas ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, start=1):
        print(f"  {i}. {col}")

    print(f"\nTarget: {target_col}")
    print(f"  Range: {y.min():.2f} - {y.max():.2f} ppm")
    print(f"  Média: {y.mean():.2f} ppm")
    print(f"  Std: {y.std():.2f} ppm")

    # Setup GroupKFold (leave-one-structure-out)
    n_splits = df["pdb_id"].nunique()
    if n_splits < 2:
        print("\nErro: precisa de pelo menos 2 estruturas (pdb_id distintos) para GroupKFold.")
        sys.exit(1)

    gkf = GroupKFold(n_splits=n_splits)

    print(f"\nValidação Cruzada: GroupKFold com {n_splits} splits")
    print("Estratégia: Leave-one-structure-out\n")

    # Alphas para RidgeCV
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

    # Armazenar resultados
    all_predictions = []
    fold_metrics = []

    # Loop por fold
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_structures = sorted(list(set(groups[train_idx])))
        test_structure = sorted(list(set(groups[test_idx])))[0]

        print(f"{'='*70}")
        print(f"FOLD {fold_idx}/{n_splits}")
        print(f"{'='*70}")
        print(f"  Treino: {train_structures} ({len(train_idx)} amostras)")
        print(f"  Teste:  [{test_structure}] ({len(test_idx)} amostras)")

        # Normalizar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Ridge com CV interna (3-fold)
        ridge = RidgeCV(alphas=alphas, cv=3, scoring="neg_mean_absolute_error")
        ridge.fit(X_train_scaled, y_train)

        best_alpha = float(ridge.alpha_)
        print(f"  Lambda escolhido: {best_alpha}")

        y_pred = ridge.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print("\n  Métricas do fold:")
        print(f"    MAE:  {mae:.3f} ppm")
        print(f"    RMSE: {rmse:.3f} ppm")
        print(f"    R²:   {r2:.3f}\n")

        fold_metrics.append(
            {
                "fold": fold_idx,
                "test_structure": test_structure,
                "train_structures": ",".join(train_structures),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "lambda": best_alpha,
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2),
            }
        )

        test_df = df.iloc[test_idx].copy()
        test_df["shift_predicted"] = y_pred
        test_df["error"] = y_test - y_pred
        test_df["abs_error"] = np.abs(test_df["error"])
        test_df["fold"] = fold_idx
        all_predictions.append(test_df)

    predictions_df = pd.concat(all_predictions, ignore_index=True)

    # Métricas globais
    y_true_global = predictions_df["shift_ca_experimental"].values
    y_pred_global = predictions_df["shift_predicted"].values

    mae_global = mean_absolute_error(y_true_global, y_pred_global)
    rmse_global = np.sqrt(mean_squared_error(y_true_global, y_pred_global))
    r2_global = r2_score(y_true_global, y_pred_global)

    print(f"{'='*70}")
    print(f"MÉTRICAS GLOBAIS (TODAS AS {len(predictions_df)} AMOSTRAS)")
    print(f"{'='*70}")
    print(f"  MAE:  {mae_global:.3f} ppm")
    print(f"  RMSE: {rmse_global:.3f} ppm")
    print(f"  R²:   {r2_global:.3f}")

    print("\nCritério de Sucesso MVP:")
    print(f"  MAE < 3.5 ppm:  {'PASS' if mae_global < 3.5 else 'FAIL'}")
    print(f"  RMSE < 4.5 ppm: {'PASS' if rmse_global < 4.5 else 'FAIL'}")
    print(f"  R² > 0.75:      {'PASS' if r2_global > 0.75 else 'FAIL'}")

    overall_success = (mae_global < 3.5) and (rmse_global < 4.5) and (r2_global > 0.75)
    print(f"\nResultado MVP: {'SUCESSO' if overall_success else 'REVISAR'}")

    # Métricas por estrutura
    print(f"\n{'='*70}")
    print("MÉTRICAS POR ESTRUTURA")
    print(f"{'='*70}")

    structure_metrics = []
    for pdb_id in sorted(predictions_df["pdb_id"].unique()):
        struct_df = predictions_df[predictions_df["pdb_id"] == pdb_id]

        mae_struct = float(struct_df["abs_error"].mean())
        rmse_struct = float(np.sqrt((struct_df["error"] ** 2).mean()))
        r2_struct = float(
            r2_score(struct_df["shift_ca_experimental"], struct_df["shift_predicted"])
        )

        print(f"\n  {pdb_id} ({len(struct_df)} amostras):")
        print(f"    MAE:  {mae_struct:.3f} ppm")
        print(f"    RMSE: {rmse_struct:.3f} ppm")
        print(f"    R²:   {r2_struct:.3f}")

        structure_metrics.append(
            {"pdb_id": pdb_id, "n_samples": int(len(struct_df)), "MAE": mae_struct, "RMSE": rmse_struct, "R2": r2_struct}
        )

    # Outliers
    print(f"\n{'='*70}")
    print("ANÁLISE DE OUTLIERS (|erro| > 5 ppm)")
    print(f"{'='*70}")

    outliers = predictions_df[predictions_df["abs_error"] > 5.0]
    outlier_cols = [
        "pdb_id",
        "residue_index",
        "residue_type",
        "shift_ca_experimental",
        "shift_predicted",
        "error",
    ]

    if len(outliers) > 0:
        print(f"\n  {len(outliers)} outliers ({len(outliers)/len(predictions_df)*100:.1f}%):\n")
        print(outliers[outlier_cols].to_string(index=False))
    else:
        print("\n  Nenhum outlier detectado.")

    # ===========================
    # Salvar resultados (results/)
    # ===========================
    print(f"\n{'='*70}")
    print("SALVANDO RESULTADOS")
    print(f"{'='*70}\n")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    predictions_path = results_dir / "predictions.csv"
    save_cols = [
        "pdb_id",
        "bmrb_id",
        "residue_index",
        "residue_type",
        "shift_ca_experimental",
        "shift_predicted",
        "error",
        "abs_error",
        "fold",
    ]
    predictions_df[save_cols].to_csv(predictions_path, index=False)
    print(f"  Predições salvas: {predictions_path}")

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_path = results_dir / "model_performance.csv"
    fold_metrics_df.to_csv(fold_path, index=False)
    print(f"  Métricas por fold: {fold_path}")

    summary_path = results_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("SUMÁRIO DO TREINAMENTO - MVP FASE 0\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Dataset: {len(predictions_df)} amostras, {df['pdb_id'].nunique()} estruturas\n")
        f.write(f"Features: {len(feature_cols)}\n")
        f.write(f"Validação: GroupKFold ({n_splits} folds)\n\n")

        f.write("MÉTRICAS GLOBAIS:\n")
        f.write(f"  MAE:  {mae_global:.3f} ppm\n")
        f.write(f"  RMSE: {rmse_global:.3f} ppm\n")
        f.write(f"  R²:   {r2_global:.3f}\n\n")

        f.write("CRITÉRIO MVP:\n")
        f.write(f"  MAE < 3.5 ppm:  {'PASS' if mae_global < 3.5 else 'FAIL'}\n")
        f.write(f"  RMSE < 4.5 ppm: {'PASS' if rmse_global < 4.5 else 'FAIL'}\n")
        f.write(f"  R² > 0.75:      {'PASS' if r2_global > 0.75 else 'FAIL'}\n\n")

        f.write(f"Resultado: {'SUCESSO' if overall_success else 'REVISAR'}\n\n")

        f.write("=" * 70 + "\n")
        f.write("MÉTRICAS POR ESTRUTURA\n")
        f.write("=" * 70 + "\n\n")

        for metric in structure_metrics:
            f.write(f"{metric['pdb_id']} ({metric['n_samples']} amostras):\n")
            f.write(f"  MAE:  {metric['MAE']:.3f} ppm\n")
            f.write(f"  RMSE: {metric['RMSE']:.3f} ppm\n")
            f.write(f"  R²:   {metric['R2']:.3f}\n\n")

        if len(outliers) > 0:
            f.write("=" * 70 + "\n")
            f.write(f"OUTLIERS (|erro| > 5 ppm): {len(outliers)}\n")
            f.write("=" * 70 + "\n\n")
            f.write(outliers[outlier_cols].to_string(index=False))

    print(f"  Sumário salvo: {summary_path}")

    # ===========================
    # Treinar modelo final (full)
    # ===========================
    print(f"\n{'='*70}")
    print("TREINANDO MODELO FINAL EM TODO O DATASET")
    print(f"{'='*70}\n")

    scaler_final = StandardScaler()
    X_scaled_final = scaler_final.fit_transform(X)

    ridge_final = RidgeCV(alphas=alphas, cv=3, scoring="neg_mean_absolute_error")
    ridge_final.fit(X_scaled_final, y)

    print(f"Lambda escolhido (full fit): {float(ridge_final.alpha_)}")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # joblib
    from joblib import dump

    dump(ridge_final, models_dir / "ridge_model.pkl")
    dump(scaler_final, models_dir / "scaler.pkl")

    print(f"Modelo salvo: {models_dir / 'ridge_model.pkl'}")
    print(f"Scaler salvo: {models_dir / 'scaler.pkl'}")

    metadata = {
        "model_type": "RidgeCV(full-fit)",
        "n_features": int(len(feature_cols)),
        "feature_names": feature_cols,
        "n_train_samples": int(len(X)),
        "n_structures": int(df["pdb_id"].nunique()),
        "mae_cv": float(mae_global),
        "rmse_cv": float(rmse_global),
        "r2_cv": float(r2_global),
        "lambda_fullfit": float(ridge_final.alpha_),
        "trained_date": datetime.now().strftime("%Y-%m-%d"),
        "version": "1.0-mvp-phase0",
        "input_csv": str(input_path).replace("\\", "/"),
    }

    metadata_path = models_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Metadata salvo: {metadata_path}")

    print(f"\n{'='*70}\n")

    return predictions_df, fold_metrics_df, mae_global, rmse_global, r2_global


if __name__ == "__main__":
    train_and_evaluate()

"""
Por Que Treinar em TODO o Dataset no Final?
Durante CV:

Cada fold treina em ~70 amostras
Apenas para avaliação, não para deployment

Modelo final:

Treina em TODAS as 95 amostras
Maximiza informação disponível
Este é o modelo que você vai distribuir/publicar

Importante: Métricas do CV (MAE=2.4 ppm) estimam o desempenho esperado do modelo final em dados novos.
"""