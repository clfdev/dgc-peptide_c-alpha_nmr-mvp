"""
Script para gerar gráficos de performance por estrutura e por tipo de resíduo
Figure 5: Performance metrics by structure and residue type

Observação: A Figure 5 vai ser na verdade a Figure 4 e Figure 4 será a Figure 5.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

print("="*70)
print("GERANDO FIGURE 5: PERFORMANCE POR ESTRUTURA E TIPO DE RESÍDUO")
print("="*70)

# Carregar dados
df_pred = pd.read_csv('results/predictions_v2.csv')

# ============================================================================
# CALCULAR MÉTRICAS POR ESTRUTURA
# ============================================================================
print("\nCalculando métricas por estrutura...")

structures = sorted(df_pred['pdb_id'].unique())
struct_metrics = []

for struct in structures:
    struct_data = df_pred[df_pred['pdb_id'] == struct]
    
    mae = struct_data['abs_error'].mean()
    rmse = np.sqrt((struct_data['error']**2).mean())
    r2 = 1 - (struct_data['error']**2).sum() / ((struct_data['shift_ca_experimental'] - struct_data['shift_ca_experimental'].mean())**2).sum()
    n_residues = len(struct_data)
    n_outliers = len(struct_data[struct_data['abs_error'] > 5])
    
    struct_metrics.append({
        'structure': struct,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'n_residues': n_residues,
        'n_outliers': n_outliers,
        'outlier_rate': n_outliers / n_residues * 100
    })

df_struct = pd.DataFrame(struct_metrics)

# ============================================================================
# CALCULAR MÉTRICAS POR TIPO DE RESÍDUO
# ============================================================================
print("Calculando métricas por tipo de resíduo...")

residue_types = sorted(df_pred['residue_type'].unique())
residue_metrics = []

for res_type in residue_types:
    res_data = df_pred[df_pred['residue_type'] == res_type]
    
    mae = res_data['abs_error'].mean()
    n_residues = len(res_data)
    n_outliers = len(res_data[res_data['abs_error'] > 5])
    
    residue_metrics.append({
        'residue_type': res_type,
        'mae': mae,
        'n_residues': n_residues,
        'n_outliers': n_outliers,
        'outlier_rate': n_outliers / n_residues * 100
    })

df_residue = pd.DataFrame(residue_metrics).sort_values('mae', ascending=False)

# ============================================================================
# CRIAR FIGURA COM 4 SUBPLOTS
# ============================================================================
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

# ============================================================================
# (a) MAE por estrutura - Bar chart
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])

x_pos = np.arange(len(structures))
colors_mae = ['#e74c3c' if mae > 4 else '#f39c12' if mae > 2.5 else '#2ecc71' 
              for mae in df_struct['mae']]

bars = ax1.bar(x_pos, df_struct['mae'], color=colors_mae, alpha=0.8, 
               edgecolor='black', linewidth=1.2)

# Linha de threshold
ax1.axhline(y=2.5, color='green', linestyle='--', linewidth=2, alpha=0.7, 
           label='Target MAE (2.5 ppm)', zorder=0)

ax1.set_xlabel('Structure (PDB ID)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Mean Absolute Error (ppm)', fontsize=14, fontweight='bold')
ax1.set_title('(a) MAE by Structure: Identifying Well-Predicted vs Poorly-Predicted Peptides', 
             fontsize=15, fontweight='bold', pad=10)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(structures, rotation=45, ha='right', fontsize=11)
ax1.legend(fontsize=12, loc='upper left')
ax1.grid(True, alpha=0.3, axis='y')

# Adicionar valores nas barras
for i, (bar, mae, n_out, n_res) in enumerate(zip(bars, df_struct['mae'], 
                                                   df_struct['n_outliers'], 
                                                   df_struct['n_residues'])):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
            f'{mae:.2f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Adicionar número de outliers embaixo
    if n_out > 0:
        ax1.text(bar.get_x() + bar.get_width()/2., 0.1,
                f'{n_out}/{n_res}',
                ha='center', va='bottom', fontsize=7, rotation=90, color='darkred')

# ============================================================================
# (b) RMSE e R² por estrutura
# ============================================================================
ax2 = fig.add_subplot(gs[1, 0])

x_pos = np.arange(len(structures))
ax2_twin = ax2.twinx()

# RMSE (barras)
bars_rmse = ax2.bar(x_pos - 0.2, df_struct['rmse'], width=0.4, 
                    color='#3498db', alpha=0.7, edgecolor='black', 
                    linewidth=1, label='RMSE')

# R² (linha)
line_r2 = ax2_twin.plot(x_pos + 0.2, df_struct['r2'], 'o-', 
                        color='#e74c3c', linewidth=2.5, markersize=8, 
                        label='R²', markeredgecolor='black', markeredgewidth=1)

# Thresholds
ax2.axhline(y=3.5, color='blue', linestyle='--', linewidth=1.5, alpha=0.5)
ax2_twin.axhline(y=0.40, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

ax2.set_xlabel('Structure (PDB ID)', fontsize=12, fontweight='bold')
ax2.set_ylabel('RMSE (ppm)', fontsize=12, fontweight='bold', color='#3498db')
ax2_twin.set_ylabel('R²', fontsize=12, fontweight='bold', color='#e74c3c')
ax2.set_title('(b) RMSE and R² by Structure', fontsize=13, fontweight='bold', pad=10)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(structures, rotation=90, ha='right', fontsize=9)
ax2.tick_params(axis='y', labelcolor='#3498db')
ax2_twin.tick_params(axis='y', labelcolor='#e74c3c')
ax2.grid(True, alpha=0.3, axis='y')

# Legendas
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

# ============================================================================
# (c) MAE por tipo de resíduo - Bar chart
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])

# Ordenar por MAE
df_residue_sorted = df_residue.sort_values('mae', ascending=False)

y_pos = np.arange(len(df_residue_sorted))
colors_res = ['#e74c3c' if mae > 4 else '#f39c12' if mae > 3 else '#2ecc71' 
              for mae in df_residue_sorted['mae']]

bars_res = ax3.barh(y_pos, df_residue_sorted['mae'], color=colors_res, 
                    alpha=0.8, edgecolor='black', linewidth=1)

ax3.axvline(x=3.0, color='orange', linestyle='--', linewidth=2, alpha=0.7, 
           label='MAE = 3.0 ppm')
ax3.axvline(x=4.0, color='red', linestyle='--', linewidth=2, alpha=0.7, 
           label='MAE = 4.0 ppm')

ax3.set_yticks(y_pos)
ax3.set_yticklabels([f"{row['residue_type']} (n={row['n_residues']})" 
                     for _, row in df_residue_sorted.iterrows()], fontsize=10)
ax3.set_xlabel('Mean Absolute Error (ppm)', fontsize=12, fontweight='bold')
ax3.set_title('(c) MAE by Residue Type: Identifying Problematic Amino Acids', 
             fontsize=13, fontweight='bold', pad=10)
ax3.legend(fontsize=10, loc='lower right')
ax3.grid(True, alpha=0.3, axis='x')

# Adicionar valores
for i, (bar, mae) in enumerate(zip(bars_res, df_residue_sorted['mae'])):
    width = bar.get_width()
    ax3.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
            f'{mae:.2f}',
            ha='left', va='center', fontsize=9, fontweight='bold')

# ============================================================================
# (d) Boxplot de erros por tipo de resíduo
# ============================================================================
ax4 = fig.add_subplot(gs[2, :])

# Preparar dados para boxplot (apenas tipos com n > 5)
residue_types_plot = [rt for rt in residue_types if len(df_pred[df_pred['residue_type'] == rt]) > 5]
error_by_residue = [df_pred[df_pred['residue_type'] == rt]['error'].values 
                    for rt in residue_types_plot]

# Cores por categoria
def get_residue_color(rt):
    if rt in ['G', 'P']:
        return '#e74c3c'  # Problemáticos
    elif rt in ['F', 'W', 'Y']:
        return '#9b59b6'  # Aromáticos
    elif rt in ['C']:
        return '#f39c12'  # Cisteína
    else:
        return '#3498db'  # Outros

colors_box = [get_residue_color(rt) for rt in residue_types_plot]

bp = ax4.boxplot(error_by_residue, 
                positions=range(len(residue_types_plot)),
                widths=0.6,
                patch_artist=True,
                showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='yellow', 
                              markeredgecolor='black', markersize=6))

# Colorir boxes
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.2)

# Linhas de referência
ax4.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Zero error')
ax4.axhline(y=5, color='red', linestyle=':', linewidth=2, alpha=0.5, label='±5 ppm outlier threshold')
ax4.axhline(y=-5, color='red', linestyle=':', linewidth=2, alpha=0.5)

ax4.set_xticks(range(len(residue_types_plot)))
labels_with_n = [f"{rt}\n(n={len(df_pred[df_pred['residue_type'] == rt])})" 
                 for rt in residue_types_plot]
ax4.set_xticklabels(labels_with_n, fontsize=10, fontweight='bold')
ax4.set_ylabel('Prediction Error (ppm)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Residue Type', fontsize=14, fontweight='bold')
ax4.set_title('(d) Error Distribution by Residue Type:\n'
             'Boxplot Analysis Revealing Systematic Biases', 
             fontsize=15, fontweight='bold', pad=5)
ax4.legend(fontsize=11, loc='upper right')
ax4.grid(True, alpha=0.3, axis='y')

# Salvar figura
plt.savefig('results/Figure5_performance_by_structure_and_residue.png', 
           dpi=300, bbox_inches='tight')
print("\n✓ Figure 5 salva: Figure5_performance_by_structure_and_residue.png")

# ============================================================================
# RESUMO ESTATÍSTICO
# ============================================================================
print("\n" + "="*70)
print("RESUMO ESTATÍSTICO")
print("="*70)

print("\nMelhores estruturas (MAE < 2.0 ppm):")
best_struct = df_struct[df_struct['mae'] < 2.0].sort_values('mae')
for _, row in best_struct.iterrows():
    print(f"  {row['structure']}: MAE = {row['mae']:.3f} ppm, R² = {row['r2']:.3f}, n={row['n_residues']}")

print("\nPiores estruturas (MAE > 4.0 ppm):")
worst_struct = df_struct[df_struct['mae'] > 4.0].sort_values('mae', ascending=False)
for _, row in worst_struct.iterrows():
    print(f"  {row['structure']}: MAE = {row['mae']:.3f} ppm, R² = {row['r2']:.3f}, "
          f"outliers = {row['n_outliers']}/{row['n_residues']} ({row['outlier_rate']:.1f}%)")

print("\nResíduos problemáticos (MAE > 4.0 ppm):")
problem_res = df_residue[df_residue['mae'] > 4.0].sort_values('mae', ascending=False)
for _, row in problem_res.iterrows():
    print(f"  {row['residue_type']}: MAE = {row['mae']:.3f} ppm, "
          f"outliers = {row['n_outliers']}/{row['n_residues']} ({row['outlier_rate']:.1f}%)")

print("\nResíduos bem preditos (MAE < 2.5 ppm):")
good_res = df_residue[df_residue['mae'] < 2.5].sort_values('mae')
for _, row in good_res.iterrows():
    print(f"  {row['residue_type']}: MAE = {row['mae']:.3f} ppm, n={row['n_residues']}")

print("\n" + "="*70)
print("FIGURA 5 GERADA COM SUCESSO!")
print("="*70)