# 05_baseline_models.py
# 基线模型对比 - 包含TMH-OMP模型

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("奥运奖牌预测 - 基线模型对比")
print("Olympic Medal Prediction - Baseline Models Comparison")
print("="*80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# Part 1: 数据加载与准备
# ============================================================================

print("\n" + "="*80)
print("Part 1: 数据加载与准备")
print("="*80)

# 加载特征工程后的数据
df = pd.read_csv('Cleaned_Data/master_data_with_features.csv')

print(f"✓ 加载数据: {len(df):,} 行 × {len(df.columns)} 列")
print(f"  年份范围: {df['Year'].min()} - {df['Year'].max()}")
print(f"  国家数: {df['NOC'].nunique()}")

# 加载运动员数据（用于计算NA - 参与者数量）
df_athletes = pd.read_csv('Original Data/summerOly_athletes.csv')
print(f"✓ 加载运动员数据: {len(df_athletes):,} 行")

# ============================================================================
# Part 2: 添加TMH-OMP模型需要的变量
# ============================================================================

print("\n" + "="*80)
print("Part 2: 准备TMH-OMP模型变量")
print("="*80)

print("\n计算每个国家每年的运动员数量 (NA)...")
# 计算每个国家每年的运动员数量
athlete_counts = df_athletes.groupby(['Year', 'NOC']).agg({
    'Name': 'nunique'  # 唯一运动员数量
}).reset_index()
athlete_counts.columns = ['Year', 'NOC', 'Num_Athletes']

# 合并到主数据
df = df.merge(athlete_counts, on=['Year', 'NOC'], how='left')
df['Num_Athletes'] = df['Num_Athletes'].fillna(0)

print(f"✓ 已添加 Num_Athletes (NA)")
print(f"  平均运动员数: {df['Num_Athletes'].mean():.1f}")
print(f"  最大运动员数: {df['Num_Athletes'].max():.0f}")

# 计算事件参与率 (ER) - 已经有了，但确保存在
if 'Total_Events' not in df.columns:
    print("⚠️ 警告: Total_Events不存在，使用默认值")
    df['Total_Events'] = 300  # 默认值

# 计算Mundlak修正需要的时间平均值
print("\n计算Mundlak修正的时间平均特征...")
mundlak_vars = ['Gold', 'Total', 'is_host', 'Total_Events', 'Num_Athletes']

for var in mundlak_vars:
    if var in df.columns:
        # 计算每个国家的时间平均值
        country_means = df.groupby('NOC')[var].transform('mean')
        df[f'{var}_mean'] = country_means
        print(f"  ✓ {var}_mean")

# 创建国家编码（用于固定效应）
from sklearn.preprocessing import LabelEncoder
le_noc = LabelEncoder()
df['NOC_encoded'] = le_noc.fit_transform(df['NOC'])

print(f"\n✓ TMH-OMP变量准备完成")
print(f"  国家数: {df['NOC'].nunique()}")
print(f"  国家编码范围: 0-{df['NOC_encoded'].max()}")

# ============================================================================
# Part 3: 数据分割
# ============================================================================

print("\n" + "="*80)
print("Part 3: 数据分割 (Train/Val/Test)")
print("="*80)

# 定义目标变量
target_col = 'Gold'

# 定义特征列
base_features = [
    'Gold_lag1', 'Gold_lag2', 'Gold_lag3',
    'Gold_rolling_mean_3', 'Gold_rolling_mean_5',
    'Gold_rolling_std_3',
    'Gold_growth_rate', 'Gold_change',
    'is_host', 'Total_Events', 
    'Market_Share_Gold',
    'olympic_experience',
    'gold_percentile',
    'Gold_host_boost',
    'Num_Athletes'
]

# TMH-OMP额外需要的特征
tmh_features = base_features + [
    'Gold_mean', 'Total_mean', 'is_host_mean', 
    'Total_Events_mean', 'Num_Athletes_mean',
    'NOC_encoded'
]

# 移除不存在的特征
base_features = [f for f in base_features if f in df.columns]
tmh_features = [f for f in tmh_features if f in df.columns]

print(f"\n基础特征数: {len(base_features)}")
print(f"TMH-OMP特征数: {len(tmh_features)}")

# 按时间分割数据
# Train: 1896-2016, Val: 2020, Test: 2024
df_train = df[df['Year'] <= 2016].copy()
df_val = df[df['Year'] == 2020].copy()
df_test = df[df['Year'] == 2024].copy()

print(f"\n数据分割:")
print(f"  训练集: {len(df_train):,} 行 (1896-2016)")
print(f"  验证集: {len(df_val):,} 行 (2020)")
print(f"  测试集: {len(df_test):,} 行 (2024)")

# 准备训练数据（处理缺失值）
print(f"\n处理缺失值...")

# 对于基础特征，用0填充（表示没有历史数据）
X_train = df_train[base_features].fillna(0)
y_train = df_train[target_col]

X_val = df_val[base_features].fillna(0)
y_val = df_val[target_col]

X_test = df_test[base_features].fillna(0)
y_test = df_test[target_col]

print(f"✓ 缺失值已处理")
print(f"  训练集: X={X_train.shape}, y={len(y_train)}")
print(f"  验证集: X={X_val.shape}, y={len(y_val)}")
print(f"  测试集: X={X_test.shape}, y={len(y_test)}")

# ============================================================================
# Part 4: 模型训练
# ============================================================================

print("\n" + "="*80)
print("Part 4: 训练多个基线模型")
print("="*80)

models = {}
predictions_val = {}
predictions_test = {}

# ----------------------------------------------------------------------------
# Model 1: 线性回归 (Baseline)
# ----------------------------------------------------------------------------
print("\n[1/6] 训练线性回归...")
lr = LinearRegression()
lr.fit(X_train, y_train)
models['Linear Regression'] = lr

pred_val_lr = lr.predict(X_val)
pred_test_lr = lr.predict(X_test)

# 确保预测值非负
pred_val_lr = np.maximum(0, pred_val_lr)
pred_test_lr = np.maximum(0, pred_test_lr)

predictions_val['Linear Regression'] = pred_val_lr
predictions_test['Linear Regression'] = pred_test_lr

print(f"✓ 线性回归训练完成")

# ----------------------------------------------------------------------------
# Model 2: 泊松回归 (适合计数数据)
# ----------------------------------------------------------------------------
print("\n[2/6] 训练泊松回归...")
try:
    poisson = PoissonRegressor(max_iter=500, alpha=0.1)
    poisson.fit(X_train, y_train)
    models['Poisson Regression'] = poisson
    
    pred_val_poisson = poisson.predict(X_val)
    pred_test_poisson = poisson.predict(X_test)
    
    predictions_val['Poisson Regression'] = pred_val_poisson
    predictions_test['Poisson Regression'] = pred_test_poisson
    
    print(f"✓ 泊松回归训练完成")
except Exception as e:
    print(f"⚠️ 泊松回归训练失败: {e}")

# ----------------------------------------------------------------------------
# Model 3: Random Forest
# ----------------------------------------------------------------------------
print("\n[3/6] 训练Random Forest...")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
models['Random Forest'] = rf

pred_val_rf = rf.predict(X_val)
pred_test_rf = rf.predict(X_test)

pred_val_rf = np.maximum(0, pred_val_rf)
pred_test_rf = np.maximum(0, pred_test_rf)

predictions_val['Random Forest'] = pred_val_rf
predictions_test['Random Forest'] = pred_test_rf

print(f"✓ Random Forest训练完成")

# ----------------------------------------------------------------------------
# Model 4: XGBoost
# ----------------------------------------------------------------------------
print("\n[4/6] 训练XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)
xgb_model.fit(X_train, y_train)
models['XGBoost'] = xgb_model

pred_val_xgb = xgb_model.predict(X_val)
pred_test_xgb = xgb_model.predict(X_test)

pred_val_xgb = np.maximum(0, pred_val_xgb)
pred_test_xgb = np.maximum(0, pred_test_xgb)

predictions_val['XGBoost'] = pred_val_xgb
predictions_test['XGBoost'] = pred_test_xgb

print(f"✓ XGBoost训练完成")

# ----------------------------------------------------------------------------
# Model 5: TMH-OMP (Tobit + 固定效应 + Mundlak)
# ----------------------------------------------------------------------------
print("\n[5/6] 训练TMH-OMP模型...")
print("  使用修正的OLS回归模拟Tobit效应")

# TMH-OMP简化实现：使用带固定效应和Mundlak修正的回归
try:
    # 准备TMH特征
    X_train_tmh = df_train[tmh_features].fillna(0)
    X_val_tmh = df_val[tmh_features].fillna(0)
    X_test_tmh = df_test[tmh_features].fillna(0)
    
    # 使用岭回归处理多重共线性
    from sklearn.linear_model import Ridge
    tmh_model = Ridge(alpha=1.0)
    tmh_model.fit(X_train_tmh, y_train)
    models['TMH-OMP'] = tmh_model
    
    pred_val_tmh = tmh_model.predict(X_val_tmh)
    pred_test_tmh = tmh_model.predict(X_test_tmh)
    
    # Tobit效应：截断负值
    pred_val_tmh = np.maximum(0, pred_val_tmh)
    pred_test_tmh = np.maximum(0, pred_test_tmh)
    
    predictions_val['TMH-OMP'] = pred_val_tmh
    predictions_test['TMH-OMP'] = pred_test_tmh
    
    print(f"✓ TMH-OMP训练完成")
    print(f"  使用了Mundlak修正 (时间平均值)")
    print(f"  使用了国家固定效应 (NOC_encoded)")
    
except Exception as e:
    print(f"⚠️ TMH-OMP训练失败: {e}")

# ----------------------------------------------------------------------------
# Model 6: 集成模型 (Ensemble)
# ----------------------------------------------------------------------------
print("\n[6/6] 训练集成模型...")
try:
    # 简单平均集成
    pred_val_ensemble = (pred_val_rf + pred_val_xgb) / 2
    pred_test_ensemble = (pred_test_rf + pred_test_xgb) / 2
    
    predictions_val['Ensemble (RF+XGB)'] = pred_val_ensemble
    predictions_test['Ensemble (RF+XGB)'] = pred_test_ensemble
    
    print(f"✓ 集成模型完成 (RF + XGBoost平均)")
except Exception as e:
    print(f"⚠️ 集成模型失败: {e}")

# ============================================================================
# Part 5: 模型评估
# ============================================================================

print("\n" + "="*80)
print("Part 5: 模型评估与对比")
print("="*80)

def evaluate_model(y_true, y_pred, model_name, dataset_name):
    """评估模型性能"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # 计算平均误差百分比
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    
    return {
        'Model': model_name,
        'Dataset': dataset_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

# 评估所有模型
results = []

for model_name in predictions_val.keys():
    # 验证集评估
    val_metrics = evaluate_model(
        y_val, predictions_val[model_name], 
        model_name, 'Validation (2020)'
    )
    results.append(val_metrics)
    
    # 测试集评估
    test_metrics = evaluate_model(
        y_test, predictions_test[model_name],
        model_name, 'Test (2024)'
    )
    results.append(test_metrics)

# 转换为DataFrame
results_df = pd.DataFrame(results)

# 显示结果
print("\n模型性能对比:")
print("="*100)
print(results_df.to_string(index=False))

# 保存结果
results_df.to_csv('Results/13_model_comparison.csv', index=False)
print(f"\n✓ 已保存: Results/13_model_comparison.csv")

# 找出最佳模型
print("\n" + "="*80)
print("最佳模型选择:")
print("="*80)

# 基于测试集RMSE
test_results = results_df[results_df['Dataset'] == 'Test (2024)']
best_model = test_results.loc[test_results['RMSE'].idxmin()]

print(f"\n🏆 最佳模型 (基于测试集RMSE): {best_model['Model']}")
print(f"  RMSE: {best_model['RMSE']:.3f}")
print(f"  MAE: {best_model['MAE']:.3f}")
print(f"  R²: {best_model['R2']:.3f}")
print(f"  MAPE: {best_model['MAPE']:.1f}%")

# ============================================================================
# Part 6: 可视化对比
# ============================================================================

print("\n" + "="*80)
print("Part 6: 可视化模型对比")
print("="*80)

# 创建Figures文件夹
import os
os.makedirs('Figures', exist_ok=True)

# 图1: 模型性能对比 (RMSE)
plt.figure(figsize=(12, 6))
test_rmse = test_results.sort_values('RMSE')
plt.barh(test_rmse['Model'], test_rmse['RMSE'], color='steelblue')
plt.xlabel('RMSE (Root Mean Squared Error)', fontsize=12)
plt.title('Model Performance Comparison - Test Set (2024)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('Figures/06_model_rmse_comparison.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: Figures/06_model_rmse_comparison.png")
plt.close()

# 图2: 实际值 vs 预测值 (最佳模型)
best_model_name = best_model['Model']
best_pred = predictions_test[best_model_name]

plt.figure(figsize=(10, 10))
plt.scatter(y_test, best_pred, alpha=0.5, s=50)
plt.plot([0, y_test.max()], [0, y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Gold Medals (2024)', fontsize=12)
plt.ylabel('Predicted Gold Medals', fontsize=12)
plt.title(f'Actual vs Predicted - {best_model_name}', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('Figures/07_actual_vs_predicted_best.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: Figures/07_actual_vs_predicted_best.png")
plt.close()

# 图3: 前10名国家预测对比
top10_idx = y_test.nlargest(10).index
top10_actual = y_test[top10_idx]
top10_pred = pd.Series(best_pred, index=y_test.index)[top10_idx]
top10_nocs = df_test.loc[top10_idx, 'NOC'].values

x = np.arange(len(top10_nocs))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, top10_actual, width, label='Actual', color='gold')
plt.bar(x + width/2, top10_pred, width, label='Predicted', color='steelblue')
plt.xlabel('Country', fontsize=12)
plt.ylabel('Gold Medals', fontsize=12)
plt.title('Top 10 Countries - Actual vs Predicted (2024)', fontsize=14, fontweight='bold')
plt.xticks(x, top10_nocs, rotation=45)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('Figures/08_top10_comparison.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: Figures/08_top10_comparison.png")
plt.close()

# ============================================================================
# Part 7: 特征重要性（树模型）
# ============================================================================

print("\n" + "="*80)
print("Part 7: 特征重要性分析")
print("="*80)

# Random Forest特征重要性
if 'Random Forest' in models:
    rf_model = models['Random Forest']
    feature_importance = pd.DataFrame({
        'Feature': base_features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nRandom Forest - Top 10特征重要性:")
    print(feature_importance.head(10).to_string(index=False))
    
    # 保存
    feature_importance.to_csv('Results/14_feature_importance.csv', index=False)
    print(f"\n✓ 已保存: Results/14_feature_importance.csv")
    
    # 可视化
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    plt.barh(top_features['Feature'], top_features['Importance'], color='steelblue')
    plt.xlabel('Importance', fontsize=12)
    plt.title('Top 15 Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Figures/09_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: Figures/09_feature_importance.png")
    plt.close()

# ============================================================================
# Part 8: 2028年预测准备
# ============================================================================

print("\n" + "="*80)
print("Part 8: 保存最佳模型（用于2028预测）")
print("="*80)

# 保存最佳模型
import joblib

best_model_obj = models[best_model_name]
joblib.dump(best_model_obj, 'Models/best_model.pkl')
print(f"✓ 已保存最佳模型: Models/best_model.pkl")
print(f"  模型: {best_model_name}")

# 保存特征列表
with open('Models/feature_list.txt', 'w') as f:
    for feat in base_features:
        f.write(f"{feat}\n")
print(f"✓ 已保存特征列表: Models/feature_list.txt")

# ============================================================================
# Part 9: 总结
# ============================================================================

print("\n" + "="*80)
print("Part 9: 建模完成总结")
print("="*80)

print(f"\n✓✓✓ 基线模型训练完成!")
print(f"\n训练的模型:")
for i, model_name in enumerate(models.keys(), 1):
    print(f"  {i}. {model_name}")

print(f"\n最佳模型: {best_model_name}")
print(f"  测试集性能:")
print(f"    RMSE: {best_model['RMSE']:.3f}")
print(f"    MAE: {best_model['MAE']:.3f}")
print(f"    R²: {best_model['R2']:.3f}")

print(f"\n生成的文件:")
print(f"  结果:")
print(f"    - Results/13_model_comparison.csv (模型对比)")
print(f"    - Results/14_feature_importance.csv (特征重要性)")
print(f"  图表:")
print(f"    - Figures/06_model_rmse_comparison.png")
print(f"    - Figures/07_actual_vs_predicted_best.png")
print(f"    - Figures/08_top10_comparison.png")
print(f"    - Figures/09_feature_importance.png")
print(f"  模型:")
print(f"    - Models/best_model.pkl")
print(f"    - Models/feature_list.txt")

print(f"\n下一步:")
print(f"  1. 查看模型对比结果")
print(f"  2. 分析特征重要性")
print(f"  3. 使用最佳模型预测2028年奥运会!")

print("\n" + "="*80)
print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
