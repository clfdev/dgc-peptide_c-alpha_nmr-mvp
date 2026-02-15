"""
Script unificado para gerar todas as figuras de resultados do modelo Ridge
Gera:
- Figure 2: Learning Curves
- Figure 3: Diagnostic Plots
- Figure 4: Residual Analysis by Structure

Obs: This order was actually inverted in the manuscript
Figure 2: Residual Analysis by Structure
Figure 4: Learning Curves
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("GERANDO TODAS AS FIGURAS DE RESULTADOS")
print("="*70)

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

# Carregar dados
print("\nCarregando dados...")
df = pd.read_csv('/mnt/project/pilot_dataset_features_v2.csv')
df_pred = pd.read_csv('/mnt/project/predictions_v2.csv')

# Features
geometric_features = ['d1', 'd2', 'd3', 'd4', 'd5', 'R_g', 'mean_dist', 
                      'dist_to_center', 'local_density', 'dist_to_n_term', 
                      'dist_to_c_term', 'max_dist']

physicochemical_features = ['hydrophobicity', 'volume', 'charge', 'polarity', 
                           'aromaticity', 'flexibility', 'accessibility',
                           'helix_propensity', 'sheet_propensity', 'turn_propensity',
                           'is_gly', 'is_pro', 'is_charged', 'is_aromatic', 'is_aliphatic']

feature_cols = geometric_features + physicochemical_features
X = df[feature_cols].values
y = df['shift_ca_experimental'].values
groups = df['pdb_id'].values

print(f"Dataset: {len(X)} amostras, {len(feature_cols)} features, {len(np.unique(groups))} estruturas")

# ============================================================================
# FIGURE 2: LEARNING CURVES
# ============================================================================
print("\n" + "="*70)
print("GERANDO FIGURE 2: LEARNING CURVES")
print("="*70)

fractions = [0.25, 0.33, 0.50, 0.67, 0.75, 0.90, 1.0]
n_splits = 21

results_lc = {
    'fraction': [], 'n_structures': [], 'n_residues_mean': [], 'n_residues_std': [],
    'mae_mean': [], 'mae_std': [], 'rmse_mean': [], 'rmse_std': [],
    'r2_mean': [], 'r2_std': []
}

for frac in fractions:
    print(f"  Processando fração: {frac*100:.0f}%")
    
    fold_maes, fold_rmses, fold_r2s, fold_n_residues = [], [], [], []
    gkf = GroupKFold(n_splits=n_splits)
    
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train_full, X_test = X[train_idx], X[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        
        unique_structures = np.unique(groups_train)
        n_structures_to_use = max(1, int(len(unique_structures) * frac))
        
        np.random.seed(42 + fold_idx)
        selected_structures = np.random.choice(unique_structures, size=n_structures_to_use, replace=False)
        
        mask = np.isin(groups_train, selected_structures)
        X_train = X_train_full[mask]
        y_train = y_train_full[mask]
        fold_n_residues.append(len(X_train))
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        fold_maes.append(mean_absolute_error(y_test, y_pred))
        fold_rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        fold_r2s.append(r2_score(y_test, y_pred))
    
    results_lc['fraction'].append(frac)
    results_lc['n_structures'].append(n_structures_to_use)
    results_lc['n_residues_mean'].append(np.mean(fold_n_residues))
    results_lc['n_residues_std'].append(np.std(fold_n_residues))
    results_lc['mae_mean'].append(np.mean(fold_maes))
    results_lc['mae_std'].append(np.std(fold_maes))
    results_lc['rmse_mean'].append(np.mean(fold_rmses))
    results_lc['rmse_std'].append(np.std(fold_rmses))
    results_lc['r2_mean'].append(np.mean(fold_r2s))
    results_lc['r2_std'].append(np.std(fold_r2s))

# Power law fit
def power_law(n, a, b):
    return a * n**(-b)

n_residues = np.array(results_lc['n_residues_mean'])
mae_values = np.array(results_lc['mae_mean'])
valid_idx = n_residues > 50
n_fit = n_residues[valid_idx]
mae_fit = mae_values[valid_idx]

try:
    popt_mae, _ = curve_fit(power_law, n_fit, mae_fit, p0=[10, 0.3])
    n_pred = np.linspace(50, 1000, 100)
    mae_pred = power_law(n_pred, *popt_mae)
    has_powerlaw = True
except:
    has_powerlaw = False

# Plotar Figure 2
fig2, axes = plt.subplots(1, 3, figsize=(18, 5))

# MAE
ax = axes[0]
ax.errorbar(results_lc['n_residues_mean'], results_lc['mae_mean'], 
            yerr=results_lc['mae_std'], fmt='o-', markersize=8, linewidth=2, 
            capsize=5, color='#e74c3c', label='Observed MAE')
if has_powerlaw:
    ax.plot(n_pred, mae_pred, '--', color='#34495e', linewidth=2, 
            label=f'Power law fit: {popt_mae[0]:.2f}N$^{{-{popt_mae[1]:.2f}}}$')
ax.axhline(y=2.5, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Target MAE (2.5 ppm)')
ax.set_xlabel('Number of Training Residues', fontsize=14, fontweight='bold')
ax.set_ylabel('Mean Absolute Error (ppm)', fontsize=14, fontweight='bold')
ax.set_title('Learning Curve: MAE vs Training Set Size', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# RMSE
ax = axes[1]
ax.errorbar(results_lc['n_residues_mean'], results_lc['rmse_mean'], 
            yerr=results_lc['rmse_std'], fmt='o-', markersize=8, linewidth=2, 
            capsize=5, color='#3498db', label='Observed RMSE')
ax.axhline(y=3.5, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Target RMSE (3.5 ppm)')
ax.set_xlabel('Number of Training Residues', fontsize=14, fontweight='bold')
ax.set_ylabel('Root Mean Squared Error (ppm)', fontsize=14, fontweight='bold')
ax.set_title('Learning Curve: RMSE vs Training Set Size', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# R²
ax = axes[2]
ax.errorbar(results_lc['n_residues_mean'], results_lc['r2_mean'], 
            yerr=results_lc['r2_std'], fmt='o-', markersize=8, linewidth=2, 
            capsize=5, color='#2ecc71', label='Observed R²')
ax.axhline(y=0.40, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Target R² (0.40)')
ax.set_xlabel('Number of Training Residues', fontsize=14, fontweight='bold')
ax.set_ylabel('R² Score', fontsize=14, fontweight='bold')
ax.set_title('Learning Curve: R² vs Training Set Size', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/Figure2_learning_curve.png', dpi=300, bbox_inches='tight')
print("✓ Figure 2 salva: Figure2_learning_curve.png")

# ============================================================================
# FIGURE 3: DIAGNOSTIC PLOTS
# ============================================================================
print("\n" + "="*70)
print("GERANDO FIGURE 3: DIAGNOSTIC PLOTS")
print("="*70)

fig3 = plt.figure(figsize=(16, 12))
gs = fig3.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# (a) Scatter: Experimental vs Predicted
ax1 = fig3.add_subplot(gs[0, :])
scatter = ax1.scatter(df_pred['shift_ca_experimental'], df_pred['shift_predicted'],
                     c=df_pred['abs_error'], cmap='RdYlGn_r', s=40, alpha=0.6,
                     edgecolors='black', linewidth=0.5)
min_val = min(df_pred['shift_ca_experimental'].min(), df_pred['shift_predicted'].min())
max_val = max(df_pred['shift_ca_experimental'].max(), df_pred['shift_predicted'].max())
ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.7, label='Identity line (y=x)')
ax1.plot([min_val, max_val], [min_val+5, max_val+5], 'r:', linewidth=1.5, alpha=0.5, label='±5 ppm error band')
ax1.plot([min_val, max_val], [min_val-5, max_val-5], 'r:', linewidth=1.5, alpha=0.5)
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Absolute Error (ppm)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Experimental Cα Shift (ppm)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Predicted Cα Shift (ppm)', fontsize=14, fontweight='bold')
ax1.set_title('(a) Experimental vs Predicted Cα Chemical Shifts', fontsize=15, fontweight='bold', pad=10)
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.3)

mae = mean_absolute_error(df_pred['shift_ca_experimental'], df_pred['shift_predicted'])
rmse = np.sqrt(mean_squared_error(df_pred['shift_ca_experimental'], df_pred['shift_predicted']))
r2 = r2_score(df_pred['shift_ca_experimental'], df_pred['shift_predicted'])
stats_text = f'MAE = {mae:.3f} ppm\nRMSE = {rmse:.3f} ppm\nR² = {r2:.3f}\nn = {len(df_pred)}'
ax1.text(0.98, 0.02, stats_text, transform=ax1.transAxes, verticalalignment='bottom',
         horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
         fontsize=11, fontweight='bold')

# (b) Residual plot
ax2 = fig3.add_subplot(gs[1, 0])
ax2.scatter(df_pred['shift_ca_experimental'], df_pred['error'], c=df_pred['abs_error'],
           cmap='RdYlGn_r', s=40, alpha=0.6, edgecolors='black', linewidth=0.5)
ax2.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.7)
ax2.axhline(y=5, color='r', linestyle=':', linewidth=1.5, alpha=0.5)
ax2.axhline(y=-5, color='r', linestyle=':', linewidth=1.5, alpha=0.5)
ax2.set_xlabel('Experimental Cα Shift (ppm)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Prediction Error (ppm)', fontsize=12, fontweight='bold')
ax2.set_title('(b) Residual Plot: Error vs Experimental Shift', fontsize=13, fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3)

# (c) Error distribution
ax3 = fig3.add_subplot(gs[1, 1])
n, bins, patches = ax3.hist(df_pred['error'], bins=40, density=True, alpha=0.7, 
                            color='steelblue', edgecolor='black')
mu, sigma = df_pred['error'].mean(), df_pred['error'].std()
x = np.linspace(df_pred['error'].min(), df_pred['error'].max(), 100)
ax3.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
         label=f'Gaussian fit\nμ={mu:.2f}, σ={sigma:.2f}')
ax3.axvline(x=mu, color='darkred', linestyle='--', linewidth=2, alpha=0.7, label=f'Mean = {mu:.2f}')
ax3.axvline(x=0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Zero error')
ax3.set_xlabel('Prediction Error (ppm)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
ax3.set_title('(c) Error Distribution with Gaussian Fit', fontsize=13, fontweight='bold', pad=10)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# (d) Q-Q plot
ax4 = fig3.add_subplot(gs[2, 0])
stats.probplot(df_pred['error'], dist="norm", plot=ax4)
ax4.get_lines()[0].set_markerfacecolor('steelblue')
ax4.get_lines()[0].set_markeredgecolor('black')
ax4.get_lines()[0].set_markersize(6)
ax4.get_lines()[0].set_alpha(0.6)
ax4.get_lines()[1].set_color('red')
ax4.get_lines()[1].set_linewidth(2)
ax4.set_xlabel('Theoretical Quantiles', fontsize=12, fontweight='bold')
ax4.set_ylabel('Sample Quantiles (Error, ppm)', fontsize=12, fontweight='bold')
ax4.set_title('(d) Q-Q Plot: Error Distribution vs Normal', fontsize=13, fontweight='bold', pad=10)
ax4.grid(True, alpha=0.3)

shapiro_stat, shapiro_p = stats.shapiro(df_pred['error'])
normality_text = f'Shapiro-Wilk test:\nW = {shapiro_stat:.4f}\np = {shapiro_p:.4e}\n'
normality_text += '(Non-normal)' if shapiro_p < 0.05 else '(Normal)'
ax4.text(0.02, 0.98, normality_text, transform=ax4.transAxes, verticalalignment='top',
         horizontalalignment='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
         fontsize=10)

# (e) Heteroscedasticity
ax5 = fig3.add_subplot(gs[2, 1])
ax5.scatter(df_pred['shift_ca_experimental'], df_pred['abs_error'], c='steelblue',
           s=40, alpha=0.6, edgecolors='black', linewidth=0.5)
ax5.axhline(y=5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Outlier threshold (5 ppm)')

sorted_idx = np.argsort(df_pred['shift_ca_experimental'])
sorted_exp = df_pred['shift_ca_experimental'].iloc[sorted_idx].values
sorted_abs_err = df_pred['abs_error'].iloc[sorted_idx].values
window_size = max(3, len(sorted_exp) // 20)
if len(sorted_exp) > window_size:
    smoothed = uniform_filter1d(sorted_abs_err, size=window_size)
    ax5.plot(sorted_exp, smoothed, 'r-', linewidth=2.5, alpha=0.8, label='Trend (smoothed)')

ax5.set_xlabel('Experimental Cα Shift (ppm)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Absolute Error (ppm)', fontsize=12, fontweight='bold')
ax5.set_title('(e) Heteroscedasticity Check: |Error| vs Experimental Shift', 
              fontsize=13, fontweight='bold', pad=10)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)

plt.savefig('/mnt/user-data/outputs/Figure3_diagnostic_plots.png', dpi=300, bbox_inches='tight')
print("✓ Figure 3 salva: Figure3_diagnostic_plots.png")

# ============================================================================
# FIGURE 4: RESIDUAL BY STRUCTURE
# ============================================================================
print("\n" + "="*70)
print("GERANDO FIGURE 4: RESIDUAL BY STRUCTURE")
print("="*70)

df_pred['residue_id'] = range(len(df_pred))
structures = sorted(df_pred['pdb_id'].unique())
n_structures = len(structures)
colors = plt.cm.tab20(np.linspace(0, 1, n_structures))
structure_colors = dict(zip(structures, colors))

fig4, axes = plt.subplots(2, 1, figsize=(20, 10))

# (a) Residual plot por residue index
ax = axes[0]
for struct in structures:
    struct_data = df_pred[df_pred['pdb_id'] == struct]
    ax.scatter(struct_data['residue_id'], struct_data['error'],
              c=[structure_colors[struct]], label=struct, s=50, alpha=0.7,
              edgecolors='black', linewidth=0.5)

ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7)
ax.axhline(y=5, color='red', linestyle=':', linewidth=2, alpha=0.5, label='±5 ppm outlier threshold')
ax.axhline(y=-5, color='red', linestyle=':', linewidth=2, alpha=0.5)
ax.set_xlabel('Residue Index (Ordered by Structure)', fontsize=14, fontweight='bold')
ax.set_ylabel('Prediction Error (Experimental - Predicted, ppm)', fontsize=14, fontweight='bold')
ax.set_title('(a) Residual Plot: Prediction Error by Residue Index, Color-Coded by Structure', 
            fontsize=16, fontweight='bold', pad=15)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, ncol=2, 
         frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3)

cumsum = 0
for struct in structures[:-1]:
    n_residues = len(df_pred[df_pred['pdb_id'] == struct])
    cumsum += n_residues
    ax.axvline(x=cumsum-0.5, color='gray', linestyle='-', linewidth=1, alpha=0.3)

# (b) Boxplot por estrutura
ax = axes[1]
error_by_structure = []
struct_labels = []
struct_positions = []

for i, struct in enumerate(structures):
    struct_data = df_pred[df_pred['pdb_id'] == struct]
    error_by_structure.append(struct_data['error'].values)
    struct_labels.append(f"{struct}\n(n={len(struct_data)})")
    struct_positions.append(i)

bp = ax.boxplot(error_by_structure, positions=struct_positions, widths=0.6,
               patch_artist=True, showmeans=True,
               meanprops=dict(marker='D', markerfacecolor='red', markersize=6))

for patch, struct in zip(bp['boxes'], structures):
    patch.set_facecolor(structure_colors[struct])
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Zero error')
ax.axhline(y=5, color='red', linestyle=':', linewidth=2, alpha=0.5, label='±5 ppm threshold')
ax.axhline(y=-5, color='red', linestyle=':', linewidth=2, alpha=0.5)
ax.set_xlabel('Structure (PDB ID)', fontsize=14, fontweight='bold')
ax.set_ylabel('Prediction Error (ppm)', fontsize=14, fontweight='bold')
ax.set_title('(b) Error Distribution by Structure: Boxplot Analysis', 
            fontsize=16, fontweight='bold', pad=15)
ax.set_xticks(struct_positions)
ax.set_xticklabels(struct_labels, rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/Figure4_residual_by_structure.png', dpi=300, bbox_inches='tight')
print("✓ Figure 4 salva: Figure4_residual_by_structure.png")

# ============================================================================
# RESUMO FINAL
# ============================================================================
print("\n" + "="*70)
print("RESUMO: TODAS AS FIGURAS GERADAS COM SUCESSO")
print("="*70)
print("\nFiguras salvas em /mnt/user-data/outputs/:")
print("  • Figure2_learning_curve.png")
print("  • Figure3_diagnostic_plots.png")
print("  • Figure4_residual_by_structure.png")
print("\nScript concluído!")
print("="*70)