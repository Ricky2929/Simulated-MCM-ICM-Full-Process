# 04_feature_engineering.py
# 通用特征工程 - 为建模准备高质量特征

import pandas as pd
import numpy as np
from datetime import datetime

print("="*80)
print("特征工程 - Feature Engineering")
print("="*80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# Part 1: 加载数据
# ============================================================================

print("\n" + "="*80)
print("Part 1: 加载数据")
print("="*80)

df = pd.read_csv('Cleaned_Data/master_data.csv')

print(f"✓ 加载数据: {len(df):,} 行 × {len(df.columns)} 列")
print(f"  年份范围: {df['Year'].min()} - {df['Year'].max()}")
print(f"  国家数: {df['NOC'].nunique()}")

# 备份原始数据
df_original = df.copy()

# 确保按国家和年份排序
df = df.sort_values(['NOC', 'Year']).reset_index(drop=True)

print(f"✓ 数据已按 NOC 和 Year 排序")

# ============================================================================
# Part 2: 滞后特征 (Lag Features)
# ============================================================================

print("\n" + "="*80)
print("Part 2: 创建滞后特征 (Lag Features)")
print("="*80)

print("\n滞后特征: 使用历史奖牌数作为预测特征")
print("  - Lag 1: 上一届奥运会的奖牌数")
print("  - Lag 2: 上上届奥运会的奖牌数")
print("  - Lag 3: 上上上届奥运会的奖牌数")

# 为主要奖牌列创建滞后特征
medal_cols = ['Gold', 'Silver', 'Bronze', 'Total']

for col in medal_cols:
    for lag in [1, 2, 3]:
        new_col = f'{col}_lag{lag}'
        df[new_col] = df.groupby('NOC')[col].shift(lag)
        print(f"  ✓ 创建: {new_col}")

# 统计缺失值
lag_cols = [col for col in df.columns if 'lag' in col]
missing_lags = df[lag_cols].isna().sum().sum()
print(f"\n滞后特征缺失值: {missing_lags:,} 个")
print(f"  原因: 国家首次参赛时没有历史数据")

# ============================================================================
# Part 3: 滚动统计特征 (Rolling Statistics)
# ============================================================================

print("\n" + "="*80)
print("Part 3: 创建滚动统计特征 (Rolling Statistics)")
print("="*80)

print("\n滚动统计: 计算过去N届的平均值、最大值、最小值")

# 定义滚动窗口
windows = [3, 5]

for col in medal_cols:
    for window in windows:
        # 滚动平均
        col_mean = f'{col}_rolling_mean_{window}'
        df[col_mean] = df.groupby('NOC')[col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        # 滚动最大值
        col_max = f'{col}_rolling_max_{window}'
        df[col_max] = df.groupby('NOC')[col].transform(
            lambda x: x.rolling(window=window, min_periods=1).max()
        )
        
        # 滚动最小值
        col_min = f'{col}_rolling_min_{window}'
        df[col_min] = df.groupby('NOC')[col].transform(
            lambda x: x.rolling(window=window, min_periods=1).min()
        )
        
        # 滚动标准差
        col_std = f'{col}_rolling_std_{window}'
        df[col_std] = df.groupby('NOC')[col].transform(
            lambda x: x.rolling(window=window, min_periods=2).std()
        )
        
        print(f"  ✓ 窗口{window}: {col_mean}, {col_max}, {col_min}, {col_std}")

# ============================================================================
# Part 4: 趋势特征 (Trend Features)
# ============================================================================

print("\n" + "="*80)
print("Part 4: 创建趋势特征 (Trend Features)")
print("="*80)

print("\n趋势特征: 衡量奖牌数的变化趋势")

for col in medal_cols:
    # 1. 增长率 (与上届相比的百分比变化)
    growth_col = f'{col}_growth_rate'
    df[growth_col] = df.groupby('NOC')[col].pct_change()
    
    # 2. 绝对变化 (与上届相比的差值)
    change_col = f'{col}_change'
    df[change_col] = df.groupby('NOC')[col].diff()
    
    # 3. 加速度 (增长率的变化)
    accel_col = f'{col}_acceleration'
    df[accel_col] = df.groupby('NOC')[growth_col].diff()
    
    print(f"  ✓ {col}: 增长率、变化量、加速度")

# 趋势方向特征
for col in medal_cols:
    # 是否增长 (1=增长, 0=持平/下降)
    df[f'{col}_is_growing'] = (df[f'{col}_change'] > 0).astype(int)
    
    # 连续增长次数
    df[f'{col}_streak'] = df.groupby('NOC')[f'{col}_is_growing'].transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )

print(f"  ✓ 趋势方向和连续性特征")

# ============================================================================
# Part 5: 交互特征 (Interaction Features)
# ============================================================================

print("\n" + "="*80)
print("Part 5: 创建交互特征 (Interaction Features)")
print("="*80)

print("\n交互特征: 组合多个特征以捕捉复杂关系")

# 1. 东道主效应加成
print("\n1. 东道主效应加成:")
for col in medal_cols:
    # 东道主 × 历史平均
    avg_col = f'{col}_rolling_mean_3'
    if avg_col in df.columns:
        interaction_col = f'{col}_host_boost'
        df[interaction_col] = df['is_host'] * df[avg_col]
        print(f"  ✓ {interaction_col} = is_host × {avg_col}")

# 2. 市场份额 × 规模效应
print("\n2. 市场份额 × 规模:")
if 'Market_Share_Gold' in df.columns and 'Total_Events' in df.columns:
    df['market_size_interaction'] = df['Market_Share_Gold'] * np.log1p(df['Total_Events'])
    print(f"  ✓ market_size_interaction = Market_Share × log(Total_Events)")

# 3. 经验 × 表现
print("\n3. 经验 × 表现:")
if 'olympic_experience' in df.columns:
    df['experience_performance'] = df['olympic_experience'] * df['Gold_rolling_mean_3']
    print(f"  ✓ experience_performance = olympic_experience × Gold_avg")

# 4. 奖牌比率特征
print("\n4. 奖牌结构特征:")
# 金牌率 (Gold / Total)
df['gold_ratio'] = df['Gold'] / (df['Total'] + 1)  # +1 避免除零
df['silver_ratio'] = df['Silver'] / (df['Total'] + 1)
df['bronze_ratio'] = df['Bronze'] / (df['Total'] + 1)
print(f"  ✓ 金银铜牌占比")

# 金银比
df['gold_silver_ratio'] = df['Gold'] / (df['Silver'] + 1)
print(f"  ✓ 金银比率")

# ============================================================================
# Part 6: 竞争强度特征 (Competition Intensity)
# ============================================================================

print("\n" + "="*80)
print("Part 6: 创建竞争强度特征 (Competition Intensity)")
print("="*80)

# 每年的竞争统计
yearly_stats = df.groupby('Year').agg({
    'Gold': ['mean', 'std', 'max'],
    'NOC': 'count'
}).reset_index()

yearly_stats.columns = ['Year', 'avg_gold_year', 'std_gold_year', 'max_gold_year', 'num_countries']

# 合并回主数据
df = df.merge(yearly_stats, on='Year', how='left')

# 相对表现 (与当年平均水平比较)
df['gold_vs_avg'] = df['Gold'] - df['avg_gold_year']
df['gold_percentile'] = df.groupby('Year')['Gold'].rank(pct=True)

print(f"  ✓ 相对于年度平均的表现")
print(f"  ✓ 年度排名百分位")

# ============================================================================
# Part 7: 特殊事件特征 (Special Events)
# ============================================================================

print("\n" + "="*80)
print("Part 7: 特殊事件特征 (Special Events)")
print("="*80)

# 首次参赛标记
df['is_first_participation'] = (df.groupby('NOC').cumcount() == 0).astype(int)

# 首次获奖标记 (首次Gold > 0)
df['is_first_medal'] = ((df['Gold'] > 0) & 
                        (df.groupby('NOC')['Gold'].shift(1).fillna(0) == 0)).astype(int)

# 回归标记 (中断后重新参赛)
df['is_comeback'] = (df['years_since_last'] > 8).astype(int)  # 超过2届

print(f"  ✓ 首次参赛/获奖标记")
print(f"  ✓ 中断回归标记")

# ============================================================================
# Part 8: 数据质量检查
# ============================================================================

print("\n" + "="*80)
print("Part 8: 数据质量检查")
print("="*80)

# 统计新特征
original_cols = set(df_original.columns)
new_cols = set(df.columns) - original_cols
new_features = sorted(list(new_cols))

print(f"\n创建的新特征数量: {len(new_features)}")
print(f"总特征数: {len(df.columns)} (原始 {len(df_original.columns)} + 新增 {len(new_features)})")

# 缺失值统计
print(f"\n缺失值统计:")
missing_counts = df[new_features].isna().sum()
missing_features = missing_counts[missing_counts > 0].sort_values(ascending=False)

if len(missing_features) > 0:
    print(f"  有缺失值的特征数: {len(missing_features)}")
    print(f"\n  前10个缺失最多的特征:")
    for feat, count in missing_features.head(10).items():
        pct = count / len(df) * 100
        print(f"    {feat}: {count:,} ({pct:.1f}%)")
else:
    print(f"  ✓ 所有新特征都没有缺失值")

# 无限值检查
inf_counts = df[new_features].apply(lambda x: np.isinf(x).sum())
inf_features = inf_counts[inf_counts > 0]

if len(inf_features) > 0:
    print(f"\n⚠️ 警告: {len(inf_features)} 个特征包含无限值")
    for feat, count in inf_features.items():
        print(f"    {feat}: {count:,}")
    
    # 替换无限值为NaN
    df[new_features] = df[new_features].replace([np.inf, -np.inf], np.nan)
    print(f"  ✓ 已将无限值替换为 NaN")

# ============================================================================
# Part 9: 保存结果
# ============================================================================

print("\n" + "="*80)
print("Part 9: 保存结果")
print("="*80)

# 保存特征工程后的数据
output_file = 'Cleaned_Data/master_data_with_features.csv'
df.to_csv(output_file, index=False)
print(f"✓ 已保存: {output_file}")
print(f"  - {len(df):,} 行")
print(f"  - {len(df.columns)} 列")

# 保存特征列表
feature_list = pd.DataFrame({
    'Feature': new_features,
    'Type': ['New'] * len(new_features)
})
feature_list.to_csv('Results/11_new_features_list.csv', index=False)
print(f"✓ 已保存特征列表: Results/11_new_features_list.csv")

# 保存特征摘要
feature_summary = pd.DataFrame({
    'Category': [
        'Lag Features',
        'Rolling Statistics',
        'Trend Features',
        'Interaction Features',
        'Competition Features',
        'Special Events'
    ],
    'Count': [
        len([f for f in new_features if 'lag' in f]),
        len([f for f in new_features if 'rolling' in f]),
        len([f for f in new_features if any(x in f for x in ['growth', 'change', 'acceleration', 'streak', 'growing'])]),
        len([f for f in new_features if any(x in f for x in ['boost', 'interaction', 'ratio'])]),
        len([f for f in new_features if any(x in f for x in ['avg_gold_year', 'std_gold_year', 'vs_avg', 'percentile'])]),
        len([f for f in new_features if any(x in f for x in ['first', 'comeback'])])
    ]
})

feature_summary.to_csv('Results/12_feature_summary.csv', index=False)
print(f"✓ 已保存特征摘要: Results/12_feature_summary.csv")

print("\n特征类别统计:")
for _, row in feature_summary.iterrows():
    print(f"  {row['Category']}: {row['Count']} 个")

# ============================================================================
# Part 10: 生成报告
# ============================================================================

print("\n" + "="*80)
print("Part 10: 特征工程完成摘要")
print("="*80)

print(f"\n✓✓✓ 特征工程完成!")
print(f"\n关键统计:")
print(f"  原始特征数: {len(df_original.columns)}")
print(f"  新增特征数: {len(new_features)}")
print(f"  最终特征数: {len(df.columns)}")
print(f"  数据记录数: {len(df):,}")

print(f"\n生成的文件:")
print(f"  - Cleaned_Data/master_data_with_features.csv (主数据)")
print(f"  - Results/11_new_features_list.csv (特征列表)")
print(f"  - Results/12_feature_summary.csv (特征摘要)")

print(f"\n下一步:")
print(f"  1. 处理缺失值 (如果需要)")
print(f"  2. 特征选择 (可选)")
print(f"  3. 开始建模!")

print("\n" + "="*80)
print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
