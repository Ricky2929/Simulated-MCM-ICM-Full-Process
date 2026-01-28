# 02_investigate_issues.py
# 数据质量问题深入调查 (Deep Investigation of Data Quality Issues)
# 目的：详细分析EDA中发现的5个主要问题，为数据清洗提供具体方案

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from collections import Counter

warnings.filterwarnings('ignore')

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# 确保输出目录存在
os.makedirs('Results', exist_ok=True)
os.makedirs('Figures', exist_ok=True)

print("="*80)
print("数据质量问题深入调查")
print("Deep Investigation of Data Quality Issues")
print("="*80)
print()

# ============================================================================
# 加载数据 (Load Data)
# ============================================================================
print("正在加载数据...")
df_athletes = pd.read_csv('Original Data/summerOly_athletes.csv')
df_medal_counts = pd.read_csv('Original Data/summerOly_medal_counts.csv')
df_hosts = pd.read_csv('Original Data/summerOly_hosts.csv')
df_programs = pd.read_csv('Original Data/summerOly_programs.csv', encoding='cp1252')
df_dictionary = pd.read_csv('Original Data/data_dictionary.csv', encoding='cp1252')
print("✓ 数据加载完成\n")

# ============================================================================
# 问题1: Medal列中的'No medal'值调查
# Issue 1: Investigation of 'No medal' values in Medal column
# ============================================================================
print("\n" + "="*80)
print("问题1: Medal列中的'No medal'值调查")
print("Issue 1: Investigation of 'No medal' values")
print("="*80)

# 统计Medal列的所有值
medal_counts = df_athletes['Medal'].value_counts(dropna=False)
print("\nMedal列值分布:")
print(medal_counts)
print(f"\n总记录数: {len(df_athletes):,}")
print(f"'No medal'记录数: {medal_counts.get('No medal', 0):,} ({medal_counts.get('No medal', 0)/len(df_athletes)*100:.2f}%)")
print(f"有奖牌记录数: {medal_counts.get('Gold', 0) + medal_counts.get('Silver', 0) + medal_counts.get('Bronze', 0):,}")

# 按年份统计No medal的比例
no_medal_by_year = df_athletes[df_athletes['Medal'] == 'No medal'].groupby('Year').size()
total_by_year = df_athletes.groupby('Year').size()
no_medal_pct_by_year = (no_medal_by_year / total_by_year * 100)

print("\n各年份'No medal'占比 (前10年和后10年):")
print("前10年:")
print(no_medal_pct_by_year.head(10))
print("\n后10年:")
print(no_medal_pct_by_year.tail(10))

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 图1: Medal值分布饼图
ax1 = axes[0]
colors = ['gold', 'silver', 'brown', 'lightgray']
medal_counts_sorted = medal_counts.reindex(['Gold', 'Silver', 'Bronze', 'No medal'])
ax1.pie(medal_counts_sorted, labels=medal_counts_sorted.index, autopct='%1.1f%%',
        colors=colors, startangle=90)
ax1.set_title('Medal Value Distribution', fontsize=14, fontweight='bold')

# 图2: No medal占比趋势
ax2 = axes[1]
ax2.plot(no_medal_pct_by_year.index, no_medal_pct_by_year.values, 
         marker='o', linewidth=2, markersize=4, color='red')
ax2.set_xlabel('Year (年份)', fontsize=12)
ax2.set_ylabel('No Medal Percentage (%)', fontsize=12)
ax2.set_title("No Medal Percentage Over Time", 
              fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(70, 90)

plt.tight_layout()
plt.savefig('Figures/05_medal_column_investigation.png', dpi=300, bbox_inches='tight')
print("\n✓ 图表已保存: Figures/05_medal_column_investigation.png")
plt.close()

# 结论
print("\n【结论】:")
print("  - 'No medal'占比约84%，这是正常的（大部分运动员未获奖牌）")
print("  - 建议: 可以保留'No medal'，也可以转换为NaN，取决于后续分析需求")
print("  - 对于获奖牌统计，直接筛选Gold/Silver/Bronze即可")

# ============================================================================
# 问题2: NOC代码不一致调查
# Issue 2: Investigation of NOC code inconsistency
# ============================================================================
print("\n" + "="*80)
print("问题2: NOC代码不一致调查 (最重要！)")
print("Issue 2: Investigation of NOC Code Inconsistency (CRITICAL!)")
print("="*80)

# 获取两个数据集中的NOC
noc_athletes = set(df_athletes['NOC'].unique())
noc_medals = set(df_medal_counts['NOC'].unique())

print(f"\nAthletes数据集中的NOC数量: {len(noc_athletes)}")
print(f"Medal Counts数据集中的NOC数量: {len(noc_medals)}")
print(f"差异: {len(noc_athletes) - len(noc_medals)}")

# 找出只在Athletes中出现的NOC
only_in_athletes = noc_athletes - noc_medals
print(f"\n只在Athletes中出现的NOC ({len(only_in_athletes)}个):")
print(sorted(only_in_athletes))

# 找出只在Medal Counts中出现的NOC
only_in_medals = noc_medals - noc_athletes
if only_in_medals:
    print(f"\n只在Medal Counts中出现的NOC ({len(only_in_medals)}个):")
    print(sorted(only_in_medals))
else:
    print("\n只在Medal Counts中出现的NOC: 无")

# 详细分析只在Athletes中出现的NOC
print("\n【详细分析】只在Athletes中出现的NOC:")
print("-" * 60)
only_athletes_analysis = []
for noc in sorted(only_in_athletes):
    noc_data = df_athletes[df_athletes['NOC'] == noc]
    years = sorted(noc_data['Year'].unique())
    total_athletes = len(noc_data)
    medals = noc_data['Medal'].value_counts()
    gold = medals.get('Gold', 0)
    silver = medals.get('Silver', 0)
    bronze = medals.get('Bronze', 0)
    total_medals = gold + silver + bronze
    
    only_athletes_analysis.append({
        'NOC': noc,
        'Years': f"{min(years)}-{max(years)}",
        'Total_Athletes': total_athletes,
        'Gold': gold,
        'Silver': silver,
        'Bronze': bronze,
        'Total_Medals': total_medals
    })
    
    print(f"\n{noc}:")
    print(f"  - 参赛年份: {years}")
    print(f"  - 运动员数: {total_athletes}")
    print(f"  - 获奖牌数: 金{gold} 银{silver} 铜{bronze} (总{total_medals})")

# 保存详细分析结果
only_athletes_df = pd.DataFrame(only_athletes_analysis)
only_athletes_df.to_csv('Results/04_NOC_only_in_athletes.csv', index=False)
print("\n✓ 详细报告已保存: Results/04_NOC_only_in_athletes.csv")

# 检查这些NOC是否真的获得了奖牌但Medal Counts中缺失
print("\n【关键发现】:")
nocs_with_medals = only_athletes_df[only_athletes_df['Total_Medals'] > 0]
if len(nocs_with_medals) > 0:
    print(f"  ⚠️ 有 {len(nocs_with_medals)} 个NOC在Athletes中获得了奖牌，但Medal Counts中没有记录！")
    print(f"     这些NOC是: {list(nocs_with_medals['NOC'])}")
    print(f"     这是数据不一致的问题，需要在清洗时处理")
else:
    print(f"  ✓ 只在Athletes中出现的{len(only_in_athletes)}个NOC都没有获得奖牌")
    print(f"     这是合理的（参赛但未获奖）")

# 检查历史国家名称问题
print("\n【检查历史国家名称变更】:")
historical_patterns = []

# 常见的历史国家代码模式
historical_keywords = ['RU', 'GER', 'YUG', 'TCH', 'URS', 'GDR', 'FRG', 'EUN', 'SCG']
for keyword in historical_keywords:
    related_nocs = [noc for noc in noc_athletes if keyword in noc]
    if related_nocs:
        print(f"  - 包含'{keyword}'的NOC: {related_nocs}")
        historical_patterns.extend(related_nocs)

# ============================================================================
# 问题3: 年份一致性调查
# Issue 3: Investigation of Year consistency
# ============================================================================
print("\n" + "="*80)
print("问题3: 年份一致性调查")
print("Issue 3: Investigation of Year Consistency")
print("="*80)

years_athletes = set(df_athletes['Year'].unique())
years_medals = set(df_medal_counts['Year'].unique())
years_hosts = set(df_hosts['Year'].unique())

print(f"\nAthletes数据集年份: {sorted(years_athletes)}")
print(f"年份数量: {len(years_athletes)}")

print(f"\nMedal Counts数据集年份: {sorted(years_medals)}")
print(f"年份数量: {len(years_medals)}")

print(f"\nHosts数据集年份: {sorted(years_hosts)}")
print(f"年份数量: {len(years_hosts)}")

# 找出差异
only_in_athletes_years = years_athletes - years_medals
only_in_medals_years = years_medals - years_athletes

if only_in_athletes_years:
    print(f"\n只在Athletes中出现的年份: {sorted(only_in_athletes_years)}")
else:
    print("\n只在Athletes中出现的年份: 无")

if only_in_medals_years:
    print(f"只在Medal Counts中出现的年份: {sorted(only_in_medals_years)}")
else:
    print("只在Medal Counts中出现的年份: 无")

# 检查是否有1906年（非正式奥运会）
if 1906 in years_athletes or 1906 in years_medals:
    print("\n⚠️ 注意: 数据中包含1906年（雅典中间奥运会），这不是正式的奥运会")
    if 1906 in years_athletes:
        athletes_1906 = len(df_athletes[df_athletes['Year'] == 1906])
        print(f"   Athletes数据集中1906年记录: {athletes_1906}")
    if 1906 in years_medals:
        medals_1906 = df_medal_counts[df_medal_counts['Year'] == 1906]
        print(f"   Medal Counts数据集中1906年记录: {len(medals_1906)}")

# 【结论】
print("\n【结论】:")
if years_athletes == years_medals:
    print("  ✓ Athletes和Medal Counts的年份完全一致")
else:
    print("  ⚠️ Athletes和Medal Counts的年份不完全一致，需要注意")

# ============================================================================
# 问题4: Programs数据集缺失值调查
# Issue 4: Investigation of Programs dataset missing values
# ============================================================================
print("\n" + "="*80)
print("问题4: Programs数据集缺失值调查")
print("Issue 4: Investigation of Programs Missing Values")
print("="*80)

# 获取年份列
year_columns = [col for col in df_programs.columns 
                if col.replace('*', '').replace('.0', '').isdigit()]

print(f"\nPrograms数据集总行数: {len(df_programs)}")
print(f"年份列数: {len(year_columns)}")

# 检查哪些行有缺失值
print("\n有缺失值的行 (Sport/Discipline):")
missing_rows = df_programs[df_programs[year_columns].isnull().any(axis=1)]
print(missing_rows[['Sport', 'Discipline', 'Code']])

# 统计每一行的缺失值数量
missing_counts_by_row = df_programs[year_columns].isnull().sum(axis=1)
print(f"\n缺失值最多的运动项目:")
top_missing = df_programs.loc[missing_counts_by_row.nlargest(5).index, 
                               ['Sport', 'Discipline']]
print(top_missing)

# 按年份统计缺失值
missing_by_year = df_programs[year_columns].isnull().sum()
print(f"\n各年份的缺失运动项目数:")
print(missing_by_year[missing_by_year > 0])

# 【结论】
print("\n【结论】:")
print("  - Programs数据集的缺失值代表该运动项目在该年份未举办")
print("  - 这是正常的，因为不是所有运动项目每届都举办")
print("  - 建议: 保持NaN或填充为0/False，表示'未举办'")

# ============================================================================
# 问题5: Dictionary数据集检查
# Issue 5: Investigation of Dictionary dataset
# ============================================================================
print("\n" + "="*80)
print("问题5: Dictionary数据集检查")
print("Issue 5: Investigation of Dictionary Dataset")
print("="*80)

print("\nDictionary数据集结构:")
print(df_dictionary.head(10))
print(f"\n形状: {df_dictionary.shape}")
print(f"\n列名: {df_dictionary.columns.tolist()}")

# 检查Unnamed列
unnamed_cols = [col for col in df_dictionary.columns if 'Unnamed' in col]
print(f"\nUnnamed列: {unnamed_cols}")
if unnamed_cols:
    print("\n这些列的内容:")
    for col in unnamed_cols:
        print(f"  {col}: {df_dictionary[col].value_counts(dropna=False).to_dict()}")

# 【结论】
print("\n【结论】:")
print("  - Dictionary文件可能有格式问题，包含多余的空列")
print("  - Unnamed列可以删除")
print("  - 此文件主要用于参考，对建模影响较小")

# ============================================================================
# 问题6: Team列特殊情况检查（新发现）
# Issue 6: Investigation of Team column special cases
# ============================================================================
print("\n" + "="*80)
print("问题6: Team列特殊情况检查（额外发现）")
print("Issue 6: Investigation of Team Column Special Cases")
print("="*80)

# 检查Team列是否包含特殊格式（如Germany-1, Germany-2）
team_examples = df_athletes['Team'].value_counts().head(20)
print("\nTeam列最常见的值:")
print(team_examples)

# 检查包含数字后缀的Team
team_with_numbers = df_athletes[df_athletes['Team'].str.contains(r'-\d+', na=False, regex=True)]
if len(team_with_numbers) > 0:
    print(f"\n发现包含数字后缀的Team: {len(team_with_numbers)} 条记录")
    print("示例:")
    print(team_with_numbers[['Name', 'Team', 'NOC', 'Year', 'Sport', 'Event']].head(10))
    
    # 统计哪些运动有这种情况
    sports_with_numbered_teams = team_with_numbers['Sport'].value_counts()
    print("\n哪些运动有数字后缀的Team:")
    print(sports_with_numbered_teams)
    
    print("\n【解释】:")
    print("  - 某些团队运动（如沙滩排球、网球双打）同一国家有多支队伍")
    print("  - Team列用 'Country-1', 'Country-2' 等区分")
    print("  - 这是正常的，在分析时需要注意")
else:
    print("\n未发现包含数字后缀的Team")

# ============================================================================
# 综合报告生成 (Comprehensive Report)
# ============================================================================
print("\n" + "="*80)
print("综合调查报告总结")
print("Comprehensive Investigation Report Summary")
print("="*80)

investigation_summary = {
    'Issue': [
        '1. Medal列"No medal"值',
        '2. NOC代码不一致 ⚠️',
        '3. 年份一致性',
        '4. Programs缺失值',
        '5. Dictionary格式问题',
        '6. Team列数字后缀'
    ],
    'Severity': [
        '低 (Low)',
        '高 (High)',
        '中 (Medium)',
        '低 (Low)',
        '低 (Low)',
        '低 (Low)'
    ],
    'Status': [
        'Understood - Keep or convert to NaN',
        'Critical - Need standardization',
        'Check if consistent',
        'Normal - Means not held',
        'Can ignore or clean',
        'Normal - Team numbering'
    ],
    'Action_Required': [
        '决定是否转换为NaN',
        '必须标准化NOC代码',
        '验证年份一致性',
        '填充为0或保持NaN',
        '删除Unnamed列',
        '理解即可，无需处理'
    ]
}

summary_df = pd.DataFrame(investigation_summary)
print("\n问题优先级和处理建议:")
print(summary_df.to_string(index=False))

# 保存综合报告
summary_df.to_csv('Results/05_investigation_summary.csv', index=False)
print("\n✓ 综合报告已保存: Results/05_investigation_summary.csv")

# 特别关注: NOC不一致问题
print("\n" + "="*80)
print("⚠️ 最需要关注的问题: NOC代码不一致")
print("="*80)
print("\n在下一步的数据清洗中，我们需要:")
print("1. 标准化NOC代码（处理历史国家名称）")
print("2. 决定如何处理只在Athletes中出现但有奖牌的NOC")
print("3. 创建NOC映射表，确保数据一致性")

# ============================================================================
# 保存关键数据用于清洗 (Save Key Data for Cleaning)
# ============================================================================

# 保存NOC列表
noc_comparison = pd.DataFrame({
    'NOC': sorted(noc_athletes | noc_medals),
    'In_Athletes': [noc in noc_athletes for noc in sorted(noc_athletes | noc_medals)],
    'In_Medals': [noc in noc_medals for noc in sorted(noc_athletes | noc_medals)]
})
noc_comparison.to_csv('Results/06_NOC_comparison.csv', index=False)
print("\n✓ NOC对比表已保存: Results/06_NOC_comparison.csv")

# 保存年份列表
years_comparison = pd.DataFrame({
    'Year': sorted(years_athletes | years_medals),
    'In_Athletes': [year in years_athletes for year in sorted(years_athletes | years_medals)],
    'In_Medals': [year in years_medals for year in sorted(years_athletes | years_medals)],
    'In_Hosts': [year in years_hosts for year in sorted(years_athletes | years_medals)]
})
years_comparison.to_csv('Results/07_years_comparison.csv', index=False)
print("✓ 年份对比表已保存: Results/07_years_comparison.csv")

print("\n" + "="*80)
print("调查完成！Investigation Completed!")
print("="*80)
print("\n生成的文件:")
print("  Results/")
print("    - 04_NOC_only_in_athletes.csv (只在Athletes中的NOC详情)")
print("    - 05_investigation_summary.csv (综合调查报告)")
print("    - 06_NOC_comparison.csv (NOC对比表)")
print("    - 07_years_comparison.csv (年份对比表)")
print("  Figures/")
print("    - 05_medal_column_investigation.png (Medal列调查)")
print("\n下一步: 根据调查结果进行数据清洗 (Data Cleaning)")
print("="*80)
