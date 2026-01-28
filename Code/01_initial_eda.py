# 01_initial_eda.py
# 初始探索性数据分析 (Initial Exploratory Data Analysis)
# 目的：理解数据结构，发现数据质量问题，为后续清洗做准备

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 创建输出目录（如果不存在）
os.makedirs('Results', exist_ok=True)
os.makedirs('Figures', exist_ok=True)

print("="*80)
print("奥运会奖牌数据 - 初始探索性数据分析")
print("Olympic Medal Data - Initial Exploratory Data Analysis")
print("="*80)
print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# Part 1: 数据加载 (Data Loading)
# ============================================================================
print("\n" + "="*80)
print("Part 1: 数据加载 (Loading Datasets)")
print("="*80)

try:
    # 加载所有5个数据集
    df_athletes = pd.read_csv('Original Data/summerOly_athletes.csv')
    print("✓ 加载 summerOly_athletes.csv 成功")
    
    df_medal_counts = pd.read_csv('Original Data/summerOly_medal_counts.csv')
    print("✓ 加载 summerOly_medal_counts.csv 成功")
    
    df_hosts = pd.read_csv('Original Data/summerOly_hosts.csv')
    print("✓ 加载 summerOly_hosts.csv 成功")
    
    df_programs = pd.read_csv('Original Data/summerOly_programs.csv', encoding='cp1252')
    print("✓ 加载 summerOly_programs.csv 成功")
    
    df_dictionary = pd.read_csv('Original Data/data_dictionary.csv', encoding='cp1252')
    print("✓ 加载 data_dictionary.csv 成功")
    
    print("\n所有数据集加载成功！")
    
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    exit()

# ============================================================================
# Part 2: 基本信息检查 (Basic Information Check)
# ============================================================================
print("\n" + "="*80)
print("Part 2: 基本信息检查 (Basic Dataset Information)")
print("="*80)

# 创建数据集概览字典
datasets = {
    'Athletes': df_athletes,
    'Medal Counts': df_medal_counts,
    'Hosts': df_hosts,
    'Programs': df_programs,
    'Dictionary': df_dictionary
}

# 显示每个数据集的基本信息
dataset_summary = []
for name, df in datasets.items():
    print(f"\n【{name}】数据集信息:")
    print(f"  - 行数 (Rows): {df.shape[0]:,}")
    print(f"  - 列数 (Columns): {df.shape[1]}")
    print(f"  - 列名 (Column names): {', '.join(df.columns.tolist())}")
    print(f"  - 内存使用 (Memory): {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    dataset_summary.append({
        'Dataset': name,
        'Rows': df.shape[0],
        'Columns': df.shape[1],
        'Memory_MB': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
    })

# 保存数据集概览
summary_df = pd.DataFrame(dataset_summary)
summary_df.to_csv('Results/01_dataset_summary.csv', index=False)
print("\n✓ 数据集概览已保存到: Results/01_dataset_summary.csv")

# ============================================================================
# Part 3: 各数据集详细检查 (Detailed Dataset Inspection)
# ============================================================================
print("\n" + "="*80)
print("Part 3: 各数据集详细检查 (Detailed Dataset Inspection)")
print("="*80)

# --- 3.1 Athletes Dataset ---
print("\n【3.1 Athletes Dataset 运动员数据集】")
print("-" * 60)
print("前5行数据:")
print(df_athletes.head())
print("\n数据类型:")
print(df_athletes.dtypes)
print("\n基本统计:")
print(df_athletes.describe(include='all'))
print(f"\n唯一值统计:")
print(f"  - 唯一运动员数: {df_athletes['Name'].nunique():,}")
print(f"  - 唯一国家数 (NOC): {df_athletes['NOC'].nunique()}")
print(f"  - 唯一年份数: {df_athletes['Year'].nunique()}")
print(f"  - 唯一运动项目数: {df_athletes['Sport'].nunique()}")
print(f"  - 唯一比赛项目数: {df_athletes['Event'].nunique()}")

# --- 3.2 Medal Counts Dataset ---
print("\n【3.2 Medal Counts Dataset 奖牌统计数据集】")
print("-" * 60)
print("前5行数据:")
print(df_medal_counts.head())
print("\n数据类型:")
print(df_medal_counts.dtypes)
print("\n基本统计:")
print(df_medal_counts.describe())
print(f"\n年份范围: {df_medal_counts['Year'].min()} - {df_medal_counts['Year'].max()}")
print(f"唯一国家数 (NOC): {df_medal_counts['NOC'].nunique()}")

# --- 3.3 Hosts Dataset ---
print("\n【3.3 Hosts Dataset 主办国数据集】")
print("-" * 60)
print("完整数据:")
print(df_hosts)
print(f"\n主办国数量: {df_hosts['Host'].nunique()}")
print(f"主办次数: {len(df_hosts)}")

# --- 3.4 Programs Dataset ---
print("\n【3.4 Programs Dataset 比赛项目数据集】")
print("-" * 60)
print("前5行数据:")
print(df_programs.head())
print(f"\n运动项目数: {len(df_programs)}")
print(f"年份列: {[col for col in df_programs.columns if col.isdigit() or '*' in str(col)]}")

# ============================================================================
# Part 4: 数据质量检查 (Data Quality Check)
# ============================================================================
print("\n" + "="*80)
print("Part 4: 数据质量检查 (Data Quality Assessment)")
print("="*80)

# --- 4.1 缺失值检查 ---
print("\n【4.1 缺失值检查 (Missing Values)】")
print("-" * 60)

missing_report = []
for name, df in datasets.items():
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    if missing.sum() > 0:
        print(f"\n{name} 数据集的缺失值:")
        for col in missing[missing > 0].index:
            print(f"  - {col}: {missing[col]} ({missing_pct[col]}%)")
            missing_report.append({
                'Dataset': name,
                'Column': col,
                'Missing_Count': missing[col],
                'Missing_Percentage': missing_pct[col]
            })
    else:
        print(f"\n{name} 数据集: 无缺失值 ✓")

if missing_report:
    missing_df = pd.DataFrame(missing_report)
    missing_df.to_csv('Results/02_missing_values_report.csv', index=False)
    print("\n✓ 缺失值报告已保存到: Results/02_missing_values_report.csv")

# --- 4.2 重复值检查 ---
print("\n【4.2 重复值检查 (Duplicate Values)】")
print("-" * 60)

for name, df in datasets.items():
    duplicates = df.duplicated().sum()
    print(f"{name}: {duplicates} 条重复记录")

# --- 4.3 Athletes数据集特殊检查 ---
print("\n【4.3 Athletes 数据集特殊检查】")
print("-" * 60)

# Medal列的值分布
print("\nMedal列值分布:")
print(df_athletes['Medal'].value_counts(dropna=False))

# Sex列的值分布
print("\nSex列值分布:")
print(df_athletes['Sex'].value_counts())

# 检查是否有异常年份
print(f"\n年份范围: {df_athletes['Year'].min()} - {df_athletes['Year'].max()}")
print(f"唯一年份: {sorted(df_athletes['Year'].unique())}")

# ============================================================================
# Part 5: 初步可视化分析 (Initial Visualizations)
# ============================================================================
print("\n" + "="*80)
print("Part 5: 初步可视化分析 (Initial Visualizations)")
print("="*80)
print("正在生成图表...")

# --- 5.1 历届奥运会奖牌总数趋势 ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图1: 每届奥运会奖牌总数
ax1 = axes[0, 0]
medals_by_year = df_medal_counts.groupby('Year')['Total'].sum().sort_index()
ax1.plot(medals_by_year.index, medals_by_year.values, marker='o', linewidth=2, markersize=6)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Total Medals', fontsize=12)
ax1.set_title('Total Medals Over Time', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 图2: 参赛国家数量趋势
ax2 = axes[0, 1]
countries_by_year = df_medal_counts.groupby('Year')['NOC'].nunique().sort_index()
ax2.plot(countries_by_year.index, countries_by_year.values, marker='s', 
         linewidth=2, markersize=6, color='green')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Number of Countries', fontsize=12)
ax2.set_title('Number of Medal-Winning Countries', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 图3: 金银铜牌分布趋势
ax3 = axes[1, 0]
gold_by_year = df_medal_counts.groupby('Year')['Gold'].sum()
silver_by_year = df_medal_counts.groupby('Year')['Silver'].sum()
bronze_by_year = df_medal_counts.groupby('Year')['Bronze'].sum()
ax3.plot(gold_by_year.index, gold_by_year.values, marker='o', label='Gold', linewidth=2)
ax3.plot(silver_by_year.index, silver_by_year.values, marker='s', label='Silver', linewidth=2)
ax3.plot(bronze_by_year.index, bronze_by_year.values, marker='^', label='Bronze', linewidth=2)
ax3.set_xlabel('Year', fontsize=12)
ax3.set_ylabel('Number of Medals', fontsize=12)
ax3.set_title('Gold/Silver/Bronze Medals Over Time', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 图4: 运动员数量趋势
ax4 = axes[1, 1]
athletes_by_year = df_athletes.groupby('Year').size()
ax4.bar(athletes_by_year.index, athletes_by_year.values, color='coral', alpha=0.7)
ax4.set_xlabel('Year', fontsize=12)
ax4.set_ylabel('Number of Athletes', fontsize=12)
ax4.set_title('Number of Athletes Over Time', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('Figures/01_olympic_trends_overview.png', dpi=300, bbox_inches='tight')
print("✓ 保存图表: Figures/01_olympic_trends_overview.png")
plt.close()

# --- 5.2 Top 10 国家奖牌数 (2024年) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 图1: 金牌Top 10
ax1 = axes[0]
top_gold_2024 = df_medal_counts[df_medal_counts['Year'] == 2024].nlargest(10, 'Gold')
ax1.barh(top_gold_2024['NOC'], top_gold_2024['Gold'], color='gold', edgecolor='black')
ax1.set_xlabel('Gold Medals ', fontsize=12)
ax1.set_ylabel('Country ', fontsize=12)
ax1.set_title('Top 10 Countries by Gold Medals (2024)', 
              fontsize=14, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')

# 图2: 总奖牌Top 10
ax2 = axes[1]
top_total_2024 = df_medal_counts[df_medal_counts['Year'] == 2024].nlargest(10, 'Total')
colors = ['gold', 'silver', 'brown']
bottom = np.zeros(len(top_total_2024))
for i, medal_type in enumerate(['Gold', 'Silver', 'Bronze']):
    ax2.barh(top_total_2024['NOC'], top_total_2024[medal_type], 
             left=bottom, label=medal_type, color=colors[i], alpha=0.8, edgecolor='black')
    bottom += top_total_2024[medal_type].values

ax2.set_xlabel('Total Medals ', fontsize=12)
ax2.set_ylabel('Country ', fontsize=12)
ax2.set_title('Top 10 Countries by Total Medals (2024)', 
              fontsize=14, fontweight='bold')
ax2.legend()
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('Figures/02_top10_countries_2024.png', dpi=300, bbox_inches='tight')
print("✓ 保存图表: Figures/02_top10_countries_2024.png")
plt.close()

# --- 5.3 运动项目数量趋势 ---
fig, ax = plt.subplots(figsize=(14, 6))

# 提取年份列（排除非年份列）
year_columns = [col for col in df_programs.columns 
                if col.replace('*', '').isdigit() and int(col.replace('*', '')) >= 1896]

# 计算每年的运动项目数（非空值）
events_by_year = {}
for year_col in year_columns:
    year = int(year_col.replace('*', ''))
    # 统计非空的项目数
    events_by_year[year] = df_programs[year_col].notna().sum()

years = sorted(events_by_year.keys())
event_counts = [events_by_year[y] for y in years]

ax.plot(years, event_counts, marker='o', linewidth=2, markersize=6, color='purple')
ax.set_xlabel('Year ', fontsize=12)
ax.set_ylabel('Number of Sports/Disciplines ', fontsize=12)
ax.set_title('Number of Sports/Disciplines Over Time', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Figures/03_sports_disciplines_trend.png', dpi=300, bbox_inches='tight')
print("✓ 保存图表: Figures/03_sports_disciplines_trend.png")
plt.close()

# --- 5.4 性别参与度分析 ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 图1: 男女运动员数量趋势
ax1 = axes[0]
gender_by_year = df_athletes.groupby(['Year', 'Sex']).size().unstack(fill_value=0)
if 'M' in gender_by_year.columns and 'F' in gender_by_year.columns:
    ax1.plot(gender_by_year.index, gender_by_year['M'], marker='o', 
             label='Male ', linewidth=2, markersize=6)
    ax1.plot(gender_by_year.index, gender_by_year['F'], marker='s', 
             label='Female ', linewidth=2, markersize=6)
    ax1.set_xlabel('Year ', fontsize=12)
    ax1.set_ylabel('Number of Athletes ', fontsize=12)
    ax1.set_title('Male vs Female Athletes Over Time', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

# 图2: 女性参与比例
ax2 = axes[1]
if 'M' in gender_by_year.columns and 'F' in gender_by_year.columns:
    total = gender_by_year['M'] + gender_by_year['F']
    female_pct = (gender_by_year['F'] / total * 100)
    ax2.plot(female_pct.index, female_pct.values, marker='o', 
             linewidth=2, markersize=6, color='red')
    ax2.set_xlabel('Year ', fontsize=12)
    ax2.set_ylabel('Female Percentage ', fontsize=12)
    ax2.set_title('Female Athletes Percentage Over Time', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 60)

plt.tight_layout()
plt.savefig('Figures/04_gender_participation.png', dpi=300, bbox_inches='tight')
print("✓ 保存图表: Figures/04_gender_participation.png")
plt.close()

# ============================================================================
# Part 6: 关键发现汇总 (Key Findings Summary)
# ============================================================================
print("\n" + "="*80)
print("Part 6: 关键发现汇总 (Key Findings Summary)")
print("="*80)

findings = []

# 发现1: 数据集大小
findings.append({
    'Category': 'Data Size',
    'Finding': f"Athletes dataset: {len(df_athletes):,} records | Medal Counts: {len(df_medal_counts):,} records"
})

# 发现2: 时间范围
findings.append({
    'Category': 'Time Range',
    'Finding': f"Athletes: {df_athletes['Year'].min()}-{df_athletes['Year'].max()} | Medals: {df_medal_counts['Year'].min()}-{df_medal_counts['Year'].max()}"
})

# 发现3: 国家数量
findings.append({
    'Category': 'Countries',
    'Finding': f"Unique NOCs in athletes: {df_athletes['NOC'].nunique()} | In medal counts: {df_medal_counts['NOC'].nunique()}"
})

# 发现4: 运动项目
findings.append({
    'Category': 'Sports',
    'Finding': f"Unique sports: {df_athletes['Sport'].nunique()} | Unique events: {df_athletes['Event'].nunique():,}"
})

# 发现5: 奖牌分布
medal_dist = df_athletes['Medal'].value_counts()
findings.append({
    'Category': 'Medal Distribution',
    'Finding': f"No medal: {medal_dist.get('No medal', 0):,} | Gold: {medal_dist.get('Gold', 0):,} | Silver: {medal_dist.get('Silver', 0):,} | Bronze: {medal_dist.get('Bronze', 0):,}"
})

# 发现6: 2024奥运会
medals_2024 = df_medal_counts[df_medal_counts['Year'] == 2024]
if len(medals_2024) > 0:
    findings.append({
        'Category': '2024 Olympics',
        'Finding': f"Countries won medals: {len(medals_2024)} | Total medals: {medals_2024['Total'].sum()}"
    })

# 发现7: 缺失值情况
total_missing = sum([df.isnull().sum().sum() for df in datasets.values()])
findings.append({
    'Category': 'Missing Values',
    'Finding': f"Total missing values across all datasets: {total_missing}"
})

# 打印发现
print("\n关键发现:")
for i, finding in enumerate(findings, 1):
    print(f"{i}. [{finding['Category']}] {finding['Finding']}")

# 保存发现
findings_df = pd.DataFrame(findings)
findings_df.to_csv('Results/03_key_findings.csv', index=False)
print("\n✓ 关键发现已保存到: Results/03_key_findings.csv")

# ============================================================================
# Part 7: 数据质量问题总结 (Data Quality Issues)
# ============================================================================
print("\n" + "="*80)
print("Part 7: 数据质量问题总结 (Data Quality Issues to Address)")
print("="*80)

issues = []

# 检查缺失值
for name, df in datasets.items():
    missing = df.isnull().sum()
    if missing.sum() > 0:
        issues.append(f"❗ {name}: 有 {missing.sum()} 个缺失值需要处理")

# 检查Medal列
if 'Medal' in df_athletes.columns:
    medal_values = df_athletes['Medal'].unique()
    if len(medal_values) > 0:
        issues.append(f"ℹ️  Medal列有以下值: {list(medal_values)}")

# 检查NOC一致性
noc_athletes = set(df_athletes['NOC'].unique())
noc_medals = set(df_medal_counts['NOC'].unique())
if noc_athletes != noc_medals:
    issues.append(f"⚠️  Athletes和Medal Counts中的NOC代码不完全一致")

# 检查年份
year_athletes = set(df_athletes['Year'].unique())
year_medals = set(df_medal_counts['Year'].unique())
if year_athletes != year_medals:
    issues.append(f"⚠️  Athletes和Medal Counts中的年份不完全一致")

if issues:
    print("\n需要在数据清洗中处理的问题:")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
else:
    print("\n✓ 未发现明显的数据质量问题")

# ============================================================================
# 结束 (Completion)
# ============================================================================
print("\n" + "="*80)
print("初始EDA完成！(Initial EDA Completed!)")
print("="*80)
print("\n生成的文件:")
print("  Results/")
print("    - 01_dataset_summary.csv")
if missing_report:
    print("    - 02_missing_values_report.csv")
print("    - 03_key_findings.csv")
print("  Figures/")
print("    - 01_olympic_trends_overview.png")
print("    - 02_top10_countries_2024.png")
print("    - 03_sports_disciplines_trend.png")
print("    - 04_gender_participation.png")
print("\n下一步: 基于以上发现，进行数据清洗和预处理")
print("="*80)
