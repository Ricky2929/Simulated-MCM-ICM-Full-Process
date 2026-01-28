# 03_data_cleaning_preprocessing.py
# 数据清洗与预处理完整脚本 (Complete Data Cleaning & Preprocessing Script)
# 2025 MCM Problem C - Olympic Medal Prediction

"""
本脚本执行以下11个步骤的数据清洗与预处理：
Step 1: 基础数据加载与检查
Step 2: 排除1906年数据
Step 3: NOC代码标准化
Step 4: 历史国家处理
Step 5: 团体项目去重
Step 6: Programs表处理
Step 7: 东道主特征创建
Step 8: 时间特征创建
Step 9: 数据合并与验证
Step 10: 市场份额计算
Step 11: 保存清洗后的数据
"""

import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# 创建输出目录
os.makedirs('Results', exist_ok=True)
os.makedirs('Cleaned_Data', exist_ok=True)

# 日志记录函数
def log(message, level="INFO"):
    """记录日志信息"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level}] {message}")

# ============================================================================
# Step 1: 基础数据加载与检查 (Basic Data Loading & Inspection)
# ============================================================================
log("="*80)
log("Step 1: 基础数据加载与检查")
log("Step 1: Basic Data Loading & Inspection")
log("="*80)

# 1.1 加载所有数据集
log("正在加载数据集...")

df_athletes = pd.read_csv('Original Data/summerOly_athletes.csv')
log(f"✓ Athletes: {df_athletes.shape[0]:,} 行, {df_athletes.shape[1]} 列")

df_medal_counts = pd.read_csv('Original Data/summerOly_medal_counts.csv')
log(f"✓ Medal Counts: {df_medal_counts.shape[0]:,} 行, {df_medal_counts.shape[1]} 列")

df_hosts = pd.read_csv('Original Data/summerOly_hosts.csv')
log(f"✓ Hosts: {df_hosts.shape[0]:,} 行, {df_hosts.shape[1]} 列")

df_programs = pd.read_csv('Original Data/summerOly_programs.csv', encoding='cp1252')
log(f"✓ Programs: {df_programs.shape[0]:,} 行, {df_programs.shape[1]} 列")

# 1.2 记录原始统计信息
original_stats = {
    'athletes_rows': len(df_athletes),
    'athletes_noc_count': df_athletes['NOC'].nunique(),
    'athletes_years': sorted(df_athletes['Year'].unique()),
    'medal_counts_rows': len(df_medal_counts),
    'medal_counts_noc_count': df_medal_counts['NOC'].nunique(),
    'gold_records_in_athletes': (df_athletes['Medal'] == 'Gold').sum()
}

log(f"原始统计 - Athletes中的Gold记录: {original_stats['gold_records_in_athletes']:,}")
log(f"原始统计 - Athletes中的唯一NOC: {original_stats['athletes_noc_count']}")
log(f"原始统计 - Medal Counts中的唯一NOC: {original_stats['medal_counts_noc_count']}")

# ============================================================================
# Step 2: 排除1906年数据 (Exclude 1906 Data)
# ============================================================================
log("\n" + "="*80)
log("Step 2: 排除1906年数据")
log("Step 2: Exclude 1906 Intercalated Games")
log("="*80)

# 1906年是雅典中间奥运会，不被IOC官方承认
before_1906 = len(df_athletes)
df_athletes = df_athletes[df_athletes['Year'] != 1906].copy()
after_1906 = len(df_athletes)

log(f"排除1906年数据: {before_1906:,} → {after_1906:,} 行 (删除 {before_1906 - after_1906:,} 行)")

# 验证
if 1906 in df_athletes['Year'].values:
    log("⚠️ 警告: 1906年数据仍然存在！", "WARNING")
else:
    log("✓ 验证通过: 1906年数据已成功排除")

# ============================================================================
# Step 3: NOC代码标准化 (NOC Code Standardization)
# ============================================================================
log("\n" + "="*80)
log("Step 3: NOC代码标准化 (P0 - 最高优先级)")
log("Step 3: NOC Code Standardization (P0 - Highest Priority)")
log("="*80)

log("问题: Athletes使用NOC代码, Medal Counts使用完整国家名称")
log("解决方案: 将Medal Counts的国家名称转换为NOC代码")

# 3.1 创建NOC到国家名称的映射字典
# 这个映射基于IOC官方标准和数据集实际情况
noc_to_country_mapping = {
    # 主要国家 (已验证的映射)
    'USA': 'United States',
    'CHN': 'China',
    'GBR': 'Great Britain',
    'FRA': 'France',
    'GER': 'Germany',
    'ITA': 'Italy',
    'JPN': 'Japan',
    'AUS': 'Australia',
    'CAN': 'Canada',
    'NED': 'Netherlands',
    'RUS': 'Russia',
    'KOR': 'South Korea',
    'ESP': 'Spain',
    'BRA': 'Brazil',
    'HUN': 'Hungary',
    'SWE': 'Sweden',
    'POL': 'Poland',
    'ROU': 'Romania',
    'CUB': 'Cuba',
    'NOR': 'Norway',
    'FIN': 'Finland',
    'BUL': 'Bulgaria',
    'DEN': 'Denmark',
    'SUI': 'Switzerland',
    'BEL': 'Belgium',
    'NZL': 'New Zealand',
    'GRE': 'Greece',
    'CZE': 'Czech Republic',
    'AUT': 'Austria',
    'UKR': 'Ukraine',
    'TUR': 'Turkey',
    'MEX': 'Mexico',
    'RSA': 'South Africa',
    'ARG': 'Argentina',
    'IRL': 'Ireland',
    'EGY': 'Egypt',
    'IND': 'India',
    'JAM': 'Jamaica',
    'KEN': 'Kenya',
    'ETH': 'Ethiopia',
    'THA': 'Thailand',
    'INA': 'Indonesia',
    'MAS': 'Malaysia',
    'PHI': 'Philippines',
    'VIE': 'Vietnam',
    'PRK': 'North Korea',
    'KAZ': 'Kazakhstan',
    'UZB': 'Uzbekistan',
    'GEO': 'Georgia',
    'AZE': 'Azerbaijan',
    'BLR': 'Belarus',
    'CRO': 'Croatia',
    'SRB': 'Serbia',
    'SLO': 'Slovenia',
    'SVK': 'Slovakia',
    'EST': 'Estonia',
    'LAT': 'Latvia',
    'LTU': 'Lithuania',
    'ARM': 'Armenia',
    'MDA': 'Moldova',
    'ALG': 'Algeria',
    'TUN': 'Tunisia',
    'MAR': 'Morocco',
    'NGR': 'Nigeria',
    'CMR': 'Cameroon',
    'ZIM': 'Zimbabwe',
    'COL': 'Colombia',
    'VEN': 'Venezuela',
    'CHI': 'Chile',
    'ECU': 'Ecuador',
    'PER': 'Peru',
    'URU': 'Uruguay',
    'DOM': 'Dominican Republic',
    'PUR': 'Puerto Rico',
    'TTO': 'Trinidad and Tobago',
    'BAH': 'Bahamas',
    'PAK': 'Pakistan',
    'SRI': 'Sri Lanka',
    'MGL': 'Mongolia',
    'TPE': 'Chinese Taipei',
    'HKG': 'Hong Kong',
    'SGP': 'Singapore',
    'POR': 'Portugal',
    'ISR': 'Israel',
    'IRI': 'Iran',
    'KSA': 'Saudi Arabia',
    'KUW': 'Kuwait',
    'QAT': 'Qatar',
    'UAE': 'United Arab Emirates',
    'JOR': 'Jordan',
    'LBN': 'Lebanon',
    'SYR': 'Syria',
    'IRQ': 'Iraq',
    'BAR': 'Barbados',
    'GRN': 'Grenada',
    'LCA': 'Saint Lucia',
    'DMA': 'Dominica',
    'SKN': 'Saint Kitts and Nevis',
    'VIN': 'Saint Vincent and the Grenadines',
    'GUY': 'Guyana',
    'SUR': 'Suriname',
    'BER': 'Bermuda',
    'ISL': 'Iceland',
    'CYP': 'Cyprus',
    'MLT': 'Malta',
    'MON': 'Monaco',
    'SMR': 'San Marino',
    'LIE': 'Liechtenstein',
    'LUX': 'Luxembourg',
    'AND': 'Andorra',
    'ALB': 'Albania',
    'MKD': 'North Macedonia',
    'BIH': 'Bosnia and Herzegovina',
    'MNE': 'Montenegro',
    'KOS': 'Kosovo',
    'AFG': 'Afghanistan',
    'BAN': 'Bangladesh',
    'BHU': 'Bhutan',
    'NEP': 'Nepal',
    'MDV': 'Maldives',
    'MYA': 'Myanmar',
    'LAO': 'Laos',
    'CAM': 'Cambodia',
    'BRU': 'Brunei',
    'TLS': 'Timor-Leste',
    'FIJ': 'Fiji',
    'PNG': 'Papua New Guinea',
    'SOL': 'Solomon Islands',
    'VAN': 'Vanuatu',
    'SAM': 'Samoa',
    'TOG': 'Togo',
    'TGA': 'Tonga',
    'KIR': 'Kiribati',
    'TUV': 'Tuvalu',
    'NRU': 'Nauru',
    'PLW': 'Palau',
    'FSM': 'Micronesia',
    'MHL': 'Marshall Islands',
    'COK': 'Cook Islands',
    'BEN': 'Benin',
    'BUR': 'Burkina Faso',
    'CPV': 'Cape Verde',
    'CIV': 'Ivory Coast',
    'GAM': 'Gambia',
    'GHA': 'Ghana',
    'GUI': 'Guinea',
    'GBS': 'Guinea-Bissau',
    'LBR': 'Liberia',
    'MLI': 'Mali',
    'MTN': 'Mauritania',
    'MRI': 'Mauritius',
    'NIG': 'Niger',
    'SEN': 'Senegal',
    'SLE': 'Sierra Leone',
    'BDI': 'Burundi',
    'COM': 'Comoros',
    'COD': 'Democratic Republic of the Congo',
    'CGO': 'Republic of the Congo',
    'DJI': 'Djibouti',
    'ERI': 'Eritrea',
    'GAB': 'Gabon',
    'LES': 'Lesotho',
    'MAD': 'Madagascar',
    'MAW': 'Malawi',
    'MOZ': 'Mozambique',
    'NAM': 'Namibia',
    'RWA': 'Rwanda',
    'STP': 'Sao Tome and Principe',
    'SEY': 'Seychelles',
    'SOM': 'Somalia',
    'SSD': 'South Sudan',
    'SUD': 'Sudan',
    'SWZ': 'Eswatini',
    'TAN': 'Tanzania',
    'UGA': 'Uganda',
    'ZAM': 'Zambia',
    'BOT': 'Botswana',
    'CAF': 'Central African Republic',
    'CHA': 'Chad',
    'GEQ': 'Equatorial Guinea',
    'ANG': 'Angola',
    'YEM': 'Yemen',
    'OMA': 'Oman',
    'BRN': 'Bahrain',
    'PLE': 'Palestine',
    'LBA': 'Libya',
    'GUA': 'Guatemala',
    'HON': 'Honduras',
    'NCA': 'Nicaragua',
    'CRC': 'Costa Rica',
    'PAN': 'Panama',
    'ESA': 'El Salvador',
    'BIZ': 'Belize',
    'HAI': 'Haiti',
    'CAY': 'Cayman Islands',
    'IVB': 'British Virgin Islands',
    'ISV': 'US Virgin Islands',
    'ANT': 'Netherlands Antilles',
    'ARU': 'Aruba',
    'BOL': 'Bolivia',
    'PAR': 'Paraguay',
    'GUM': 'Guam',
    'ASA': 'American Samoa',
    
    # 历史国家/特殊代码 (将在Step 4中处理)
    # 这里先保留原始映射，不做转换
    'ANZ': 'Australasia',  # 澳大拉西亚 (1908-1912)
    'EOR': 'Refugee Olympic Team',  # 难民队
    'IOA': 'Independent Olympic Athletes',
    'BOH': 'Bohemia',  # 波西米亚 (历史)
}

# 3.2 创建反向映射 (Country Name → NOC)
country_to_noc_mapping = {v: k for k, v in noc_to_country_mapping.items()}

# 3.3 标准化Medal Counts数据集
log("正在标准化Medal Counts数据集的NOC代码...")

# 创建NOC列
df_medal_counts['NOC_original'] = df_medal_counts['NOC'].copy()  # 保存原始值用于验证

# 转换国家名称为NOC代码
df_medal_counts['NOC'] = df_medal_counts['NOC'].map(country_to_noc_mapping)

# 检查是否有未映射的国家
unmapped = df_medal_counts[df_medal_counts['NOC'].isna()]['NOC_original'].unique()
if len(unmapped) > 0:
    log(f"⚠️ 警告: 有 {len(unmapped)} 个国家名称未能映射到NOC:", "WARNING")
    for country in unmapped[:10]:  # 只显示前10个
        log(f"    - {country}")
else:
    log("✓ 所有国家名称已成功映射到NOC代码")

# 3.4 验证
noc_athletes = set(df_athletes['NOC'].unique())
noc_medals = set(df_medal_counts['NOC'].dropna().unique())
only_in_athletes = noc_athletes - noc_medals
only_in_medals = noc_medals - noc_athletes

log(f"验证 - Athletes中的NOC: {len(noc_athletes)}")
log(f"验证 - Medal Counts中的NOC: {len(noc_medals)}")
log(f"验证 - 只在Athletes中: {len(only_in_athletes)}")
log(f"验证 - 只在Medal Counts中: {len(only_in_medals)}")

# ============================================================================
# Step 4: 历史国家处理 (Historical Countries Handling)
# ============================================================================
log("\n" + "="*80)
log("Step 4: 历史国家处理 (P1 - 核心问题)")
log("Step 4: Historical Countries Handling (P1 - Core Issue)")
log("="*80)

log("采用4类分类处理框架:")
log("  情况A: 单纯改名 (Name Change)")
log("  情况B: 分裂与继承 (Dissolution & Succession)")
log("  情况C: 合并 (Merger)")
log("  情况D: 特殊马甲 (Sanctions/Special Codes)")

# 4.1 情况A: 单纯改名 (Name Change)
name_changes = {
    'CEY': 'SRI',  # Ceylon (锡兰) -> Sri Lanka (斯里兰卡)
    'RHO': 'ZIM',  # Rhodesia (罗得西亚) -> Zimbabwe (津巴布韦)
    'BUR': 'BFA',  # Upper Volta -> Burkina Faso
    'DAH': 'BEN',  # Dahomey -> Benin
    'ZAI': 'COD',  # Zaire -> Democratic Republic of the Congo
}

log("情况A - 单纯改名:")
for old_noc, new_noc in name_changes.items():
    count_athletes = (df_athletes['NOC'] == old_noc).sum()
    count_medals = (df_medal_counts['NOC'] == old_noc).sum()
    if count_athletes > 0 or count_medals > 0:
        log(f"  {old_noc} → {new_noc}: Athletes={count_athletes}, Medals={count_medals}")
        df_athletes.loc[df_athletes['NOC'] == old_noc, 'NOC'] = new_noc
        df_medal_counts.loc[df_medal_counts['NOC'] == old_noc, 'NOC'] = new_noc

# 4.2 情况B: 分裂与继承 (Dissolution & Succession)
log("\n情况B - 分裂与继承:")

# B.1 苏联 → 俄罗斯
log("  处理: URS (苏联) + EUN (独联体) → RUS (俄罗斯)")
urs_count = (df_athletes['NOC'] == 'URS').sum()
eun_count = (df_athletes['NOC'] == 'EUN').sum()
log(f"    URS记录: {urs_count:,}")
log(f"    EUN记录: {eun_count:,}")

df_athletes.loc[df_athletes['NOC'] == 'URS', 'NOC'] = 'RUS'
df_athletes.loc[df_athletes['NOC'] == 'EUN', 'NOC'] = 'RUS'
df_medal_counts.loc[df_medal_counts['NOC'] == 'URS', 'NOC'] = 'RUS'
df_medal_counts.loc[df_medal_counts['NOC'] == 'EUN', 'NOC'] = 'RUS'

# B.2 捷克斯洛伐克 → 捷克 + 斯洛伐克
log("  处理: TCH (捷克斯洛伐克) → CZE (65%) + SVK (35%)")
tch_medals = df_medal_counts[df_medal_counts['NOC'] == 'TCH'].copy()
if len(tch_medals) > 0:
    # 为捷克创建记录 (65%)
    tch_cze = tch_medals.copy()
    tch_cze['NOC'] = 'CZE'
    tch_cze['Gold'] = (tch_cze['Gold'] * 0.65).round().astype(int)
    tch_cze['Silver'] = (tch_cze['Silver'] * 0.65).round().astype(int)
    tch_cze['Bronze'] = (tch_cze['Bronze'] * 0.65).round().astype(int)
    tch_cze['Total'] = tch_cze['Gold'] + tch_cze['Silver'] + tch_cze['Bronze']
    
    # 为斯洛伐克创建记录 (35%)
    tch_svk = tch_medals.copy()
    tch_svk['NOC'] = 'SVK'
    tch_svk['Gold'] = (tch_svk['Gold'] * 0.35).round().astype(int)
    tch_svk['Silver'] = (tch_svk['Silver'] * 0.35).round().astype(int)
    tch_svk['Bronze'] = (tch_svk['Bronze'] * 0.35).round().astype(int)
    tch_svk['Total'] = tch_svk['Gold'] + tch_svk['Silver'] + tch_svk['Bronze']
    
    # 删除原TCH记录，添加新记录
    df_medal_counts = df_medal_counts[df_medal_counts['NOC'] != 'TCH']
    df_medal_counts = pd.concat([df_medal_counts, tch_cze, tch_svk], ignore_index=True)
    log(f"    TCH历史记录已分配给CZE和SVK")

# Athletes表中的TCH
df_athletes.loc[df_athletes['NOC'] == 'TCH', 'NOC'] = 'CZE'  # 简化处理，全部归入CZE

# B.3 南斯拉夫 → 塞尔维亚
log("  处理: YUG (南斯拉夫) + SCG (塞黑) → SRB (塞尔维亚)")
df_athletes.loc[df_athletes['NOC'] == 'YUG', 'NOC'] = 'SRB'
df_athletes.loc[df_athletes['NOC'] == 'SCG', 'NOC'] = 'SRB'
df_medal_counts.loc[df_medal_counts['NOC'] == 'YUG', 'NOC'] = 'SRB'
df_medal_counts.loc[df_medal_counts['NOC'] == 'SCG', 'NOC'] = 'SRB'

# 4.3 情况C: 合并 (Merger)
log("\n情况C - 合并:")
log("  处理: GDR (东德) + FRG (西德) → GER (德国) [1990年前]")

# 对于Medal Counts，需要合并1990年前的数据
gdr_medals = df_medal_counts[df_medal_counts['NOC'] == 'GDR'].copy()
frg_medals = df_medal_counts[df_medal_counts['NOC'] == 'FRG'].copy()

if len(gdr_medals) > 0 and len(frg_medals) > 0:
    # 找出共同年份
    common_years = set(gdr_medals['Year']) & set(frg_medals['Year'])
    log(f"    GDR和FRG共同参赛的年份: {sorted(common_years)}")
    
    # 为这些年份创建合并的GER记录
    for year in common_years:
        gdr_year = gdr_medals[gdr_medals['Year'] == year]
        frg_year = frg_medals[frg_medals['Year'] == year]
        
        if len(gdr_year) > 0 and len(frg_year) > 0:
            # 创建合并记录
            merged = gdr_year.iloc[0].copy()
            merged['NOC'] = 'GER'
            merged['Gold'] = gdr_year['Gold'].values[0] + frg_year['Gold'].values[0]
            merged['Silver'] = gdr_year['Silver'].values[0] + frg_year['Silver'].values[0]
            merged['Bronze'] = gdr_year['Bronze'].values[0] + frg_year['Bronze'].values[0]
            merged['Total'] = merged['Gold'] + merged['Silver'] + merged['Bronze']
            
            # 删除GDR和FRG的记录，添加GER
            df_medal_counts = df_medal_counts[~((df_medal_counts['NOC'].isin(['GDR', 'FRG'])) & 
                                                  (df_medal_counts['Year'] == year))]
            df_medal_counts = pd.concat([df_medal_counts, merged.to_frame().T], ignore_index=True)

# Athletes表中的GDR和FRG都映射到GER
df_athletes.loc[df_athletes['NOC'] == 'GDR', 'NOC'] = 'GER'
df_athletes.loc[df_athletes['NOC'] == 'FRG', 'NOC'] = 'GER'

# 剩余的GDR和FRG记录也映射到GER
df_medal_counts.loc[df_medal_counts['NOC'] == 'GDR', 'NOC'] = 'GER'
df_medal_counts.loc[df_medal_counts['NOC'] == 'FRG', 'NOC'] = 'GER'

# 4.4 情况D: 特殊马甲 (Sanctions/Special Codes)
special_codes_to_rus = {
    'ROC': 'RUS',  # Russian Olympic Committee (2020)
    'OAR': 'RUS',  # Olympic Athletes from Russia (2018, 主要是冬奥)
    'AIN': 'RUS',  # Individual Neutral Athletes (2024)
}

log("\n情况D - 特殊马甲:")
for special, target in special_codes_to_rus.items():
    count_athletes = (df_athletes['NOC'] == special).sum()
    count_medals = (df_medal_counts['NOC'] == special).sum()
    if count_athletes > 0 or count_medals > 0:
        log(f"  {special} → {target}: Athletes={count_athletes}, Medals={count_medals}")
        df_athletes.loc[df_athletes['NOC'] == special, 'NOC'] = target
        df_medal_counts.loc[df_medal_counts['NOC'] == special, 'NOC'] = target

# 4.5 验证历史国家处理
log("\n历史国家处理验证:")
historical_nocs = ['URS', 'EUN', 'GDR', 'FRG', 'TCH', 'YUG', 'SCG', 'ROC', 'OAR', 'AIN']
for noc in historical_nocs:
    count = (df_athletes['NOC'] == noc).sum()
    if count > 0:
        log(f"  ⚠️ {noc} 仍存在 {count} 条记录", "WARNING")
    else:
        log(f"  ✓ {noc} 已完全处理")

# ============================================================================
# Step 5: 团体项目去重 (Team Sports Deduplication)
# ============================================================================
log("\n" + "="*80)
log("Step 5: 团体项目去重 (P0 - 最高优先级)")
log("Step 5: Team Sports Deduplication (P0 - Highest Priority)")
log("="*80)

log("问题: 团体项目的每个成员都有一行记录，导致重复计数")
log("解决方案: 基于 ['Year', 'Event', 'NOC', 'Medal'] 去重")

# 5.1 去重前的统计
before_dedup = len(df_athletes)
gold_before = (df_athletes['Medal'] == 'Gold').sum()
silver_before = (df_athletes['Medal'] == 'Silver').sum()
bronze_before = (df_athletes['Medal'] == 'Bronze').sum()

log(f"去重前 - 总记录: {before_dedup:,}")
log(f"去重前 - Gold记录: {gold_before:,}")
log(f"去重前 - Silver记录: {silver_before:,}")
log(f"去重前 - Bronze记录: {bronze_before:,}")

# 5.2 执行去重
df_medals_dedup = df_athletes.drop_duplicates(
    subset=['Year', 'Event', 'NOC', 'Medal']
).copy()

# 5.3 去重后的统计
after_dedup = len(df_medals_dedup)
gold_after = (df_medals_dedup['Medal'] == 'Gold').sum()
silver_after = (df_medals_dedup['Medal'] == 'Silver').sum()
bronze_after = (df_medals_dedup['Medal'] == 'Bronze').sum()

log(f"去重后 - 总记录: {after_dedup:,}")
log(f"去重后 - Gold记录: {gold_after:,}")
log(f"去重后 - Silver记录: {silver_after:,}")
log(f"去重后 - Bronze记录: {bronze_after:,}")
log(f"去重效果: 删除了 {before_dedup - after_dedup:,} 条重复记录 ({(before_dedup - after_dedup)/before_dedup*100:.1f}%)")

# 5.4 关键验证: 与Medal Counts对比
log("\n关键验证: 将去重后的Athletes与Medal Counts对比")

# 计算Athletes表中2024年的奖牌数
athletes_2024 = df_medals_dedup[df_medals_dedup['Year'] == 2024]
athletes_2024_gold = (athletes_2024['Medal'] == 'Gold').sum()
athletes_2024_silver = (athletes_2024['Medal'] == 'Silver').sum()
athletes_2024_bronze = (athletes_2024['Medal'] == 'Bronze').sum()
athletes_2024_total = athletes_2024_gold + athletes_2024_silver + athletes_2024_bronze

# Medal Counts中2024年的数据
medals_2024 = df_medal_counts[df_medal_counts['Year'] == 2024]
medals_2024_gold = medals_2024['Gold'].sum()
medals_2024_silver = medals_2024['Silver'].sum()
medals_2024_bronze = medals_2024['Bronze'].sum()
medals_2024_total = medals_2024['Total'].sum()

log(f"2024年奖牌数对比:")
log(f"  Athletes (去重后) - Gold: {athletes_2024_gold}, Silver: {athletes_2024_silver}, Bronze: {athletes_2024_bronze}, Total: {athletes_2024_total}")
log(f"  Medal Counts      - Gold: {medals_2024_gold}, Silver: {medals_2024_silver}, Bronze: {medals_2024_bronze}, Total: {medals_2024_total}")

# 计算差异
diff_gold = abs(athletes_2024_gold - medals_2024_gold)
diff_total = abs(athletes_2024_total - medals_2024_total)

if diff_gold <= 10 and diff_total <= 30:  # 允许小的差异（可能由于数据记录方式）
    log(f"✓ 验证通过: 差异在可接受范围内 (Gold差异: {diff_gold}, Total差异: {diff_total})")
else:
    log(f"⚠️ 警告: 差异较大 (Gold差异: {diff_gold}, Total差异: {diff_total})", "WARNING")

# 5.5 保存去重后的Athletes数据
log("\n保存去重后的Athletes数据...")
df_medals_dedup.to_csv('Cleaned_Data/athletes_deduplicated.csv', index=False)
log("✓ 已保存: Cleaned_Data/athletes_deduplicated.csv")

# 继续使用去重后的数据
df_athletes = df_medals_dedup.copy()

log("\n" + "="*80)
log("✓ Step 1-5 完成")
log("继续执行 Step 6-11...")
log("="*80)

# 保存检查点
df_athletes.to_csv('Cleaned_Data/checkpoint_after_step5.csv', index=False)
df_medal_counts.to_csv('Cleaned_Data/checkpoint_medal_counts_after_step5.csv', index=False)
log("✓ 检查点已保存")
