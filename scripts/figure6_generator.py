import matplotlib.pyplot as plt
import numpy as np

# Dados
structures = ['1L2Y', '1H3H', '2CEF', '5U9V']
mae_values = [4.03, 1.92, 2.07, 2.27]
rmse_values = [5.53, 2.36, 2.21, 3.00]

# Cores baseadas em performance
# Verde: excelente (MAE < 2.5 e RMSE < 3.5)
# Laranja: moderado (MAE >= 2.5 OU RMSE >= 3.5, mas não ambos ruins)
# Vermelho: pobre (MAE > 4.0 ou RMSE > 5.0)
colors = []
for mae, rmse in zip(mae_values, rmse_values):
    if mae < 2.5 and rmse < 3.5:
        colors.append(('#90EE90', '#2E8B57'))  # Verde claro e escuro
    elif mae > 4.0 or rmse > 5.0:
        colors.append(('#FF6B6B', '#DC143C'))  # Vermelho claro e escuro
    else:
        colors.append(('#FFD700', '#FFA500'))  # Laranja claro e escuro

# Criar figura
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(structures))
width = 0.35

# Plotar barras
bars_mae = []
bars_rmse = []
for i, (mae, rmse, (color_mae, color_rmse)) in enumerate(zip(mae_values, rmse_values, colors)):
    bar_mae = ax.bar(i - width/2, mae, width, label='MAE' if i == 0 else '', 
                     color=color_mae, edgecolor='black', linewidth=1.5)
    bar_rmse = ax.bar(i + width/2, rmse, width, label='RMSE' if i == 0 else '', 
                      color=color_rmse, edgecolor='black', linewidth=1.5)
    bars_mae.append(bar_mae)
    bars_rmse.append(bar_rmse)

# Linhas de threshold
ax.axhline(y=2.5, color='green', linestyle='--', linewidth=2, alpha=0.7, 
          label='Target MAE (2.5 ppm)', zorder=0)
ax.axhline(y=3.5, color='blue', linestyle='--', linewidth=2, alpha=0.7, 
          label='Target RMSE (3.5 ppm)', zorder=0)

# Adicionar valores acima das barras
for i, (mae, rmse) in enumerate(zip(mae_values, rmse_values)):
    ax.text(i - width/2, mae + 0.15, f'{mae:.2f}', 
           ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(i + width/2, rmse + 0.15, f'{rmse:.2f}', 
           ha='center', va='bottom', fontsize=11, fontweight='bold')

# Configurações do gráfico
ax.set_xlabel('Structure', fontsize=13, fontweight='bold')
ax.set_ylabel('Error (ppm)', fontsize=13, fontweight='bold')
ax.set_title('MAE and RMSE Comparison', fontsize=15, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(structures, fontsize=12, fontweight='bold')
ax.set_ylim(0, 6)
ax.legend(fontsize=11, loc='upper right', frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.5)

# Ajustar layout
plt.tight_layout()

# Salvar figura
plt.savefig('results/Figure6_post_training_comparison.png', 
           dpi=300, bbox_inches='tight')
print("✓ Figure 6 saved: Figure6_post_training_comparison.png")

plt.show()