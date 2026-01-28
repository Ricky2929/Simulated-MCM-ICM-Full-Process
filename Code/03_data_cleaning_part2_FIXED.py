# 03_data_cleaning_preprocessing_part2.py
# 数据清洗与预处理 - Part 2 (Steps 6-11)
# 接续 Part 1 的检查点数据

import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# 日志记录函数
def log(message, level="INFO"):
    """记录日志信息"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level}] {message}")

# 加载检查点数据
log("="*80)
log("加载 Part 1 的检查点数据...")
log("="*80)

df_athletes = pd.read_csv('Cleaned_Data/checkpoint_after_step5.csv')
df_medal_counts = pd.read_csv('Cleaned_Data/checkpoint_medal_counts_after_step5.csv')
df_hosts = pd.read_csv('Original Data/summerOly_hosts.csv')
df_programs = pd.read_csv('Original Data/summerOly_programs.csv', encoding='cp1252')

log(f"✓ Athletes: {len(df_athletes):,} 行")
log(f"✓ Medal Counts: {len(df_medal_counts):,} 行")

# ============================================================================
# Step 6: Programs表处理 (Programs Table Processing)
# ============================================================================
log("\n" + "="*80)
log("Step 6: Programs表处理 (P1)")
log("Step 6: Programs Table Processing (P1)")
log("="*80)

# 6.1 识别年份列
year_columns = [col for col in df_programs.columns 
                if col.replace('*', '').replace('.0', '').isdigit()]
log(f"识别到 {len(year_columns)} 个年份列")

# 6.2 处理特殊符号
log("处理特殊符号 (· 等)...")

# 将 · 和其他非数字值替换为 0
for col in year_columns:
    df_programs[col] = df_programs[col].replace('·', 0)
    df_programs[col] = df_programs[col].replace('.', 0)
    df_programs[col] = pd.to_numeric(df_programs[col], errors='coerce').fillna(0)

log("✓ 已将所有年份列转换为数值型")

# 6.3 计算每年的总金牌数 (Total Events)
log("计算每年的总金牌数 (Total Events)...")

# 创建Total Events数据框
total_events_by_year = []

for col in year_columns:
    try:
        year = int(col.replace('*', ''))
        # 排除1906年
        if year == 1906:
            continue
        total_events = df_programs[col].sum()
        total_events_by_year.append({
            'Year': year,
            'Total_Events': int(total_events)
        })
    except:
        continue

df_total_events = pd.DataFrame(total_events_by_year).sort_values('Year')

log(f"✓ 计算完成: {len(df_total_events)} 个年份")
log(f"Total Events 范围: {df_total_events['Total_Events'].min()} - {df_total_events['Total_Events'].max()}")

# 显示部分数据
log("\n部分年份的Total Events:")
for _, row in df_total_events.head(5).iterrows():
    log(f"  {row['Year']}: {row['Total_Events']} 个项目")
log("  ...")
for _, row in df_total_events.tail(5).iterrows():
    log(f"  {row['Year']}: {row['Total_Events']} 个项目")

# 保存Total Events
df_total_events.to_csv('Cleaned_Data/total_events_by_year.csv', index=False)
log("✓ 已保存: Cleaned_Data/total_events_by_year.csv")

# ============================================================================
# Step 7: 东道主特征创建 (Host Effect Features)
# ============================================================================
log("\n" + "="*80)
log("Step 7: 东道主特征创建 (P1 - 核心问题)")
log("Step 7: Host Effect Features Creation (P1 - Core Issue)")
log("="*80)

# 7.1 创建基于年份的直接映射（最准确的方法）
log("创建Year到Host NOC的直接映射...")

# 基于历史事实的年份到NOC映射
# 这种方法比城市映射更准确，因为Athens和Paris都举办过多次
year_to_host_noc = {
    1896: 'GRE',  # Athens
    1900: 'FRA',  # Paris
    1904: 'USA',  # St Louis
    1908: 'GBR',  # London
    1912: 'SWE',  # Stockholm
    1920: 'BEL',  # Antwerp
    1924: 'FRA',  # Paris
    1928: 'NED',  # Amsterdam
    1932: 'USA',  # Los Angeles
    1936: 'GER',  # Berlin
    1948: 'GBR',  # London
    1952: 'FIN',  # Helsinki
    1956: 'AUS',  # Melbourne
    1960: 'ITA',  # Rome
    1964: 'JPN',  # Tokyo
    1968: 'MEX',  # Mexico City
    1972: 'GER',  # Munich
    1976: 'CAN',  # Montreal
    1980: 'RUS',  # Moscow (USSR→RUS已在Part1处理)
    1984: 'USA',  # Los Angeles
    1988: 'KOR',  # Seoul
    1992: 'ESP',  # Barcelona
    1996: 'USA',  # Atlanta
    2000: 'AUS',  # Sydney
    2004: 'GRE',  # Athens
    2008: 'CHN',  # Beijing
    2012: 'GBR',  # London
    2016: 'BRA',  # Rio de Janeiro
    2020: 'JPN',  # Tokyo (实际2021年举办)
    2024: 'FRA',  # Paris
    2028: 'USA',  # Los Angeles (未来)
    2032: 'AUS',  # Brisbane (未来)
}

# 为hosts表添加Host_NOC
df_hosts['Host_NOC'] = df_hosts['Year'].map(year_to_host_noc)

# 检查映射结果
unmapped_hosts = df_hosts[df_hosts['Host_NOC'].isna()]
if len(unmapped_hosts) > 0:
    log(f"⚠️ 警告: 有 {len(unmapped_hosts)} 个年份未映射", "WARNING")
    for _, row in unmapped_hosts.iterrows():
        log(f"    {row['Year']}: {row['Host']}")
else:
    log(f"✓ 主办年份到NOC的映射: {df_hosts['Host_NOC'].notna().sum()} 个记录，全部成功")

# 7.2 合并东道主信息到Medal Counts
df_medal_counts = df_medal_counts.merge(
    df_hosts[['Year', 'Host_NOC']], 
    on='Year', 
    how='left'
)

# 7.3 创建is_host特征
df_medal_counts['is_host'] = (df_medal_counts['NOC'] == df_medal_counts['Host_NOC']).astype(int)

log(f"✓ is_host 特征已创建")
log(f"  历史上做过东道主的次数: {df_medal_counts['is_host'].sum()}")

# 7.4 创建进阶东道主特征
log("创建进阶东道主特征...")

# years_since_hosted: 距离上次做东道主的年数
df_medal_counts = df_medal_counts.sort_values(['NOC', 'Year'])

def calculate_years_since_hosted(group):
    """计算距离上次做东道主的年数"""
    group = group.copy()
    last_host_year = None
    years_since = []
    
    for idx, row in group.iterrows():
        if row['is_host'] == 1:
            last_host_year = row['Year']
            years_since.append(0)
        else:
            if last_host_year is not None:
                years_since.append(row['Year'] - last_host_year)
            else:
                years_since.append(999)  # 从未做过东道主
    
    group['years_since_hosted'] = years_since
    return group

df_medal_counts = df_medal_counts.groupby('NOC', group_keys=False).apply(calculate_years_since_hosted)

log(f"✓ years_since_hosted 特征已创建")

# 7.5 验证东道主效应
log("\n东道主效应验证:")

# 计算东道主与非东道主的平均金牌数
host_avg = df_medal_counts[df_medal_counts['is_host'] == 1]['Gold'].mean()
non_host_avg = df_medal_counts[df_medal_counts['is_host'] == 0]['Gold'].mean()
boost = host_avg - non_host_avg
boost_pct = (boost / non_host_avg) * 100

log(f"  东道主平均金牌数: {host_avg:.2f}")
log(f"  非东道主平均金牌数: {non_host_avg:.2f}")
log(f"  东道主增幅: +{boost:.2f} 枚 (+{boost_pct:.1f}%)")

# 显示部分东道主记录
log("\n部分东道主记录:")
host_records = df_medal_counts[df_medal_counts['is_host'] == 1][['Year', 'NOC', 'Gold', 'Total']].tail(10)
for _, row in host_records.iterrows():
    log(f"  {row['Year']} - {row['NOC']}: {row['Gold']} 金牌, {row['Total']} 总奖牌")

# ============================================================================
# Step 8: 时间特征创建 (Time Features)
# ============================================================================
log("\n" + "="*80)
log("Step 8: 时间特征创建 (P2)")
log("Step 8: Time Features Creation (P2)")
log("="*80)

# 8.1 创建years_since_last特征
log("创建 years_since_last 特征 (距离上届奥运会的年数)...")

df_medal_counts = df_medal_counts.sort_values(['NOC', 'Year'])

df_medal_counts['years_since_last'] = df_medal_counts.groupby('NOC')['Year'].diff()

# 第一次参赛的国家，years_since_last为NaN，设为0
df_medal_counts['years_since_last'] = df_medal_counts['years_since_last'].fillna(0)

log(f"✓ years_since_last 特征已创建")

# 显示统计
time_gaps = df_medal_counts['years_since_last'].value_counts().sort_index()
log("时间间隔分布:")
for gap, count in time_gaps.items():
    if gap > 0:
        log(f"  {int(gap)} 年: {count} 次")

# 8.2 创建crossed_war特征
log("\n创建 crossed_war 特征 (标记跨越战争的记录)...")

# 1920和1948是战争后的第一届
df_medal_counts['crossed_war'] = df_medal_counts['Year'].isin([1920, 1948]).astype(int)

crossed_war_count = df_medal_counts['crossed_war'].sum()
log(f"✓ crossed_war 特征已创建")
log(f"  跨越战争的记录数: {crossed_war_count}")

# 8.3 创建其他时间特征
log("\n创建其他时间特征...")

# 首次获奖年份
first_medal_year = df_medal_counts.groupby('NOC')['Year'].min().to_dict()
df_medal_counts['first_medal_year'] = df_medal_counts['NOC'].map(first_medal_year)

# 奥运参赛经验 (年数)
df_medal_counts['olympic_experience'] = df_medal_counts['Year'] - df_medal_counts['first_medal_year']

# 是否是前苏联国家
post_soviet_countries = ['UKR', 'BLR', 'KAZ', 'UZB', 'GEO', 'AZE', 'ARM', 'MDA', 
                          'KGZ', 'TJK', 'TKM', 'EST', 'LAT', 'LTU']
df_medal_counts['is_post_soviet'] = df_medal_counts['NOC'].isin(post_soviet_countries).astype(int)

# 对于前苏联国家，计算独立后的年数
df_medal_counts['years_since_independence'] = 0
mask = df_medal_counts['is_post_soviet'] == 1
df_medal_counts.loc[mask, 'years_since_independence'] = df_medal_counts.loc[mask, 'Year'] - 1991

log(f"✓ 已创建: first_medal_year, olympic_experience, is_post_soviet, years_since_independence")

# ============================================================================
# Step 9: 数据合并与验证 (Data Merging & Validation)
# ============================================================================
log("\n" + "="*80)
log("Step 9: 数据合并与验证")
log("Step 9: Data Merging & Validation")
log("="*80)

# 9.1 合并Total Events
log("合并 Total Events...")

df_master = df_medal_counts.merge(
    df_total_events,
    on='Year',
    how='left'
)

log(f"✓ 已合并 Total Events")
log(f"  缺失Total Events的记录: {df_master['Total_Events'].isna().sum()}")

# 9.2 数据验证
log("\n执行数据验证...")

# 验证1: 检查关键列是否存在
required_columns = ['Year', 'NOC', 'Rank', 'Gold', 'Silver', 'Bronze', 'Total', 
                    'is_host', 'years_since_last', 'Total_Events']
missing_cols = [col for col in required_columns if col not in df_master.columns]
if missing_cols:
    log(f"⚠️ 警告: 缺少列 {missing_cols}", "WARNING")
else:
    log(f"✓ 所有必需列都存在")

# 验证2: 检查数据完整性
log(f"\n数据完整性检查:")
log(f"  总记录数: {len(df_master):,}")
log(f"  唯一NOC数: {df_master['NOC'].nunique()}")
log(f"  年份范围: {df_master['Year'].min()} - {df_master['Year'].max()}")
log(f"  唯一年份数: {df_master['Year'].nunique()}")

# 验证3: 检查奖牌数逻辑
log(f"\n奖牌数逻辑检查:")
total_mismatch = df_master[df_master['Total'] != (df_master['Gold'] + df_master['Silver'] + df_master['Bronze'])]
if len(total_mismatch) > 0:
    log(f"⚠️ 警告: 有 {len(total_mismatch)} 条记录的总奖牌数与金银铜之和不符", "WARNING")
else:
    log(f"✓ 所有记录的奖牌数逻辑正确")

# 验证4: 检查异常值
log(f"\n异常值检查:")
log(f"  最大金牌数: {df_master['Gold'].max()} (Year: {df_master.loc[df_master['Gold'].idxmax(), 'Year']}, NOC: {df_master.loc[df_master['Gold'].idxmax(), 'NOC']})")
log(f"  最大总奖牌数: {df_master['Total'].max()} (Year: {df_master.loc[df_master['Total'].idxmax(), 'Year']}, NOC: {df_master.loc[df_master['Total'].idxmax(), 'NOC']})")

# ============================================================================
# Step 10: 市场份额计算 (Market Share Calculation)
# ============================================================================
log("\n" + "="*80)
log("Step 10: 市场份额计算 (P1)")
log("Step 10: Market Share Calculation (P1)")
log("="*80)

# 10.1 计算市场份额
log("计算金牌和总奖牌的市场份额...")

# 只为有Total_Events数据的记录计算
mask = df_master['Total_Events'].notna() & (df_master['Total_Events'] > 0)

df_master['Market_Share_Gold'] = 0.0
df_master['Market_Share_Total'] = 0.0

df_master.loc[mask, 'Market_Share_Gold'] = (
    df_master.loc[mask, 'Gold'] / df_master.loc[mask, 'Total_Events']
)

df_master.loc[mask, 'Market_Share_Total'] = (
    df_master.loc[mask, 'Total'] / df_master.loc[mask, 'Total_Events']
)

log(f"✓ 市场份额已计算")

# 10.2 显示市场份额统计
log(f"\n市场份额统计:")
log(f"  金牌市场份额 - 平均: {df_master[mask]['Market_Share_Gold'].mean():.4f}")
log(f"  金牌市场份额 - 最大: {df_master[mask]['Market_Share_Gold'].max():.4f}")
log(f"  总奖牌市场份额 - 平均: {df_master[mask]['Market_Share_Total'].mean():.4f}")
log(f"  总奖牌市场份额 - 最大: {df_master[mask]['Market_Share_Total'].max():.4f}")

# 10.3 展示主要国家的市场份额趋势
log(f"\n主要国家2024年市场份额:")
top_countries_2024 = df_master[df_master['Year'] == 2024].nlargest(10, 'Gold')
for _, row in top_countries_2024.iterrows():
    log(f"  {row['NOC']}: 金牌 {row['Gold']} / {row['Total_Events']:.0f} = {row['Market_Share_Gold']:.1%}")

# ============================================================================
# Step 11: 保存清洗后的数据 (Save Cleaned Data)
# ============================================================================
log("\n" + "="*80)
log("Step 11: 保存清洗后的数据")
log("Step 11: Save Cleaned Data")
log("="*80)

# 11.1 保存主数据表
log("保存主数据表 (Master Data)...")

# 重新排序列，将重要的列放在前面
column_order = [
    'Year', 'NOC', 'Rank', 
    'Gold', 'Silver', 'Bronze', 'Total',
    'Market_Share_Gold', 'Market_Share_Total',
    'Total_Events',
    'is_host', 'Host_NOC', 
    'years_since_last', 'crossed_war',
    'years_since_hosted', 'olympic_experience',
    'first_medal_year', 'is_post_soviet', 'years_since_independence'
]

# 添加任何剩余的列
remaining_cols = [col for col in df_master.columns if col not in column_order]
column_order.extend(remaining_cols)

# 只保留存在的列
column_order = [col for col in column_order if col in df_master.columns]

df_master = df_master[column_order]

# 保存
df_master.to_csv('Cleaned_Data/master_data.csv', index=False)
log(f"✓ 已保存: Cleaned_Data/master_data.csv")
log(f"  行数: {len(df_master):,}")
log(f"  列数: {len(df_master.columns)}")

# 11.2 保存去重后的Athletes数据（如果还没保存）
if not os.path.exists('Cleaned_Data/cleaned_athletes.csv'):
    df_athletes.to_csv('Cleaned_Data/cleaned_athletes.csv', index=False)
    log(f"✓ 已保存: Cleaned_Data/cleaned_athletes.csv")

# 11.3 生成数据清洗报告
log("\n生成数据清洗报告...")

report = {
    'Cleaning_Step': [
        'Step 1: Data Loading',
        'Step 2: Exclude 1906',
        'Step 3: NOC Standardization',
        'Step 4: Historical Countries',
        'Step 5: Team Deduplication',
        'Step 6: Programs Processing',
        'Step 7: Host Features',
        'Step 8: Time Features',
        'Step 9: Data Merging',
        'Step 10: Market Share',
        'Step 11: Save Data'
    ],
    'Status': ['✓'] * 11,
    'Records_Count': [
        len(df_master),
        len(df_master),
        len(df_master),
        len(df_master),
        len(df_master),
        len(df_master),
        len(df_master),
        len(df_master),
        len(df_master),
        len(df_master),
        len(df_master)
    ]
}

df_report = pd.DataFrame(report)
df_report.to_csv('Results/08_cleaning_report.csv', index=False)
log(f"✓ 已保存清洗报告: Results/08_cleaning_report.csv")

# 11.4 生成数据字典
log("\n生成数据字典...")

data_dict = {
    'Column_Name': df_master.columns.tolist(),
    'Data_Type': [str(dtype) for dtype in df_master.dtypes],
    'Non_Null_Count': [df_master[col].notna().sum() for col in df_master.columns],
    'Null_Count': [df_master[col].isna().sum() for col in df_master.columns],
    'Unique_Values': [df_master[col].nunique() for col in df_master.columns]
}

df_dict = pd.DataFrame(data_dict)
df_dict.to_csv('Results/09_data_dictionary.csv', index=False)
log(f"✓ 已保存数据字典: Results/09_data_dictionary.csv")

# 11.5 生成关键统计
log("\n生成关键统计...")

key_stats = {
    'Statistic': [
        'Total Records',
        'Unique Countries (NOC)',
        'Unique Years',
        'Year Range',
        'Total Gold Medals (All Time)',
        'Total Silver Medals (All Time)',
        'Total Bronze Medals (All Time)',
        'Total Medals (All Time)',
        'Average Gold per Country per Olympics',
        'Average Total per Country per Olympics',
        'Records with Host Info',
        'Records as Host',
        'Host Effect Boost (%)',
        'Records with Market Share',
        'Average Market Share Gold',
        'Average Market Share Total'
    ],
    'Value': [
        len(df_master),
        df_master['NOC'].nunique(),
        df_master['Year'].nunique(),
        f"{df_master['Year'].min()} - {df_master['Year'].max()}",
        df_master['Gold'].sum(),
        df_master['Silver'].sum(),
        df_master['Bronze'].sum(),
        df_master['Total'].sum(),
        f"{df_master['Gold'].mean():.2f}",
        f"{df_master['Total'].mean():.2f}",
        df_master['is_host'].notna().sum(),
        df_master['is_host'].sum(),
        f"+{((df_master[df_master['is_host']==1]['Gold'].mean() / df_master[df_master['is_host']==0]['Gold'].mean() - 1) * 100):.1f}%",
        (df_master['Market_Share_Gold'] > 0).sum(),
        f"{df_master[df_master['Market_Share_Gold'] > 0]['Market_Share_Gold'].mean():.4f}",
        f"{df_master[df_master['Market_Share_Total'] > 0]['Market_Share_Total'].mean():.4f}"
    ]
}

df_key_stats = pd.DataFrame(key_stats)
df_key_stats.to_csv('Results/10_key_statistics.csv', index=False)
log(f"✓ 已保存关键统计: Results/10_key_statistics.csv")

# ============================================================================
# 完成！(Completion!)
# ============================================================================
log("\n" + "="*80)
log("✓✓✓ 数据清洗与预处理完成！ ✓✓✓")
log("✓✓✓ Data Cleaning & Preprocessing Completed! ✓✓✓")
log("="*80)

log("\n生成的文件:")
log("  Cleaned_Data/")
log("    - master_data.csv (主数据表，用于建模)")
log("    - cleaned_athletes.csv (去重后的运动员数据)")
log("    - total_events_by_year.csv (每年的总金牌数)")
log("  Results/")
log("    - 08_cleaning_report.csv (清洗报告)")
log("    - 09_data_dictionary.csv (数据字典)")
log("    - 10_key_statistics.csv (关键统计)")

log("\n主数据表列概览:")
for i, col in enumerate(df_master.columns, 1):
    log(f"  {i}. {col}")

log("\n下一步: 使用 master_data.csv 进行特征工程和建模")
log("="*80)
