# 完整流程指南 - 从头到尾
# Complete Workflow Guide - From Scratch

## 📋 项目结构（清理后）

```
Simulated MCM&ICM Full Process/
│
├── Original Data/              ✅ 保留（5个CSV文件）
│   ├── summerOly_athletes.csv
│   ├── summerOly_medal_counts.csv
│   ├── summerOly_hosts.csv
│   ├── summerOly_programs.csv
│   └── data_dictionary.csv
│
├── Code/                       ✅ 保留（5个Python脚本）
│   ├── 01_initial_eda.py
│   ├── 02_investigate_issues.py
│   ├── 03_data_cleaning_part1.py
│   ├── 03_data_cleaning_part2_FIXED.py
│   └── create_complete_mapping.py
│
├── Cleaned_Data/               🗑️ 清空（重新生成）
│   └── (将被脚本自动创建)
│
├── Results/                    🗑️ 清空（重新生成）
│   └── (将被脚本自动创建)
│
└── Figures/                    🗑️ 清空（重新生成）
    └── (将被脚本自动创建)
```

---

## 🧹 第一步：彻底清理

### 方法A：运行清理脚本（推荐）

```bash
python clean_start.py
```

**这个脚本会：**
- ✅ 清空 Cleaned_Data/ 文件夹
- ✅ 清空 Results/ 文件夹
- ✅ 清空 Figures/ 文件夹
- ✅ 保留 Original Data/ 和 Code/
- ✅ 验证核心文件是否存在

### 方法B：手动清理

**Windows:**
```cmd
rmdir /s /q "Cleaned_Data"
rmdir /s /q "Results"
rmdir /s /q "Figures"
mkdir "Cleaned_Data"
mkdir "Results"
mkdir "Figures"
```

**Mac/Linux:**
```bash
rm -rf Cleaned_Data/*
rm -rf Results/*
rm -rf Figures/*
```

---

## 🚀 第二步：完整流程（5步）

### **阶段1: EDA（探索性数据分析）**

#### **步骤1: 初步EDA（2分钟）**

```bash
python Code/01_initial_eda.py
```

**这个脚本做什么：**
- 加载5个数据集
- 显示基本信息（行数、列数、数据类型）
- 检查缺失值
- 生成基本统计

**预期输出：**
```
✓ 加载了5个数据集
✓ Athletes: 252,565 行
✓ Medal Counts: 1,435 行
✓ Hosts: 35 行
✓ Programs: 74 行
```

**生成的文件：**
- `Results/01_dataset_summary.csv` - 数据集摘要
- 终端输出：详细统计信息

**验证点：**
- [ ] 5个数据集全部成功加载
- [ ] 没有报错信息
- [ ] Results文件夹中有输出文件

---

#### **步骤2: 深度问题调查（2-3分钟）**

```bash
python Code/02_investigate_issues.py
```

**这个脚本做什么：**
- 深入分析数据质量问题
- 识别NOC不一致
- 检查团体项目重复
- 分析历史国家
- 检查缺失值模式

**预期输出：**
```
✓ 发现7个核心问题：
  1. NOC不一致（Athletes用代码，Medal Counts用名称）
  2. 团体项目重复（13,002条Gold记录 vs 实际346枚）
  3. 历史国家需要合并
  4. 东道主效应需要建模
  5. Olympic规模变化（43枚 → 329枚）
  6. 战争年份取消
  7. Medal列有"No medal"值
```

**生成的文件：**
- `Results/02_data_quality_report.csv` - 质量报告
- `Results/03_noc_analysis.csv` - NOC分析
- `Results/04_team_sports_analysis.csv` - 团体项目分析
- `Results/05_historical_countries.csv` - 历史国家分析

**验证点：**
- [ ] 识别出主要数据质量问题
- [ ] Results中有4个新文件
- [ ] 理解每个问题的性质

---

### **阶段2: 数据清洗（Data Cleaning）**

#### **步骤3: 数据清洗 Part 1（1-2分钟）**

```bash
python Code/03_data_cleaning_part1.py
```

**这个脚本做什么：**
- Step 1: 加载数据
- Step 2: 排除1906年数据（不被IOC承认）
- Step 3: 标准化NOC代码
- Step 4: 处理历史国家
  - URS/EUN → RUS (苏联/独联体 → 俄罗斯)
  - GDR+FRG → GER (东德+西德 → 德国)
  - TCH → CZE/SVK (捷克斯洛伐克 → 捷克/斯洛伐克)
  - YUG → SRB (南斯拉夫 → 塞尔维亚)
- Step 5: 团体项目去重（核心！）

**预期输出：**
```
✓ 排除1906年: XXX → YYY 行
✓ NOC标准化完成
✓ 历史国家处理:
  ✓ URS → RUS
  ✓ GDR+FRG → GER
  ✓ TCH → CZE
  ✓ YUG → SRB
✓ 团体项目去重:
  去重前: 13,002+ Gold记录
  去重后: 346 Gold记录 (2024年)
  与Medal Counts对比: 差异 < 15枚 ✓
```

**生成的文件：**
- `Cleaned_Data/checkpoint_after_step5.csv` - 检查点
- `Cleaned_Data/athletes_deduplicated.csv` - 去重后数据

**关键验证点：**
- [ ] 2024年Gold记录 ≈ 346（不是13,002！）
- [ ] Athletes与Medal Counts的2024年差异 < 15枚
- [ ] 历史NOC全部处理完成

---

#### **步骤4: NOC映射修复（1分钟）**

```bash
python Code/create_complete_mapping.py
```

**这个脚本做什么：**
- 从Athletes自动学习NOC到国家名称的映射
- 添加250+个手动映射（所有常见国家）
- 处理历史国家名称（Russian Empire, Czechoslovakia等）
- 将Medal Counts的国家名称转换为NOC代码

**预期输出：**
```
✓ 自动学习了 233 个映射
✓ 最终映射字典包含 250+ 个映射
✓ 成功映射: 153/164 (93.3%)
✓ 未能映射: 11/164 (6.7%)

验证结果:
  ✓ 共同的NOC: 143
  ✓ 只在Athletes中: 91
  ✓ 只在Medal Counts中: 11
  
✓✓✓ 映射成功！共同NOC数 > 130
```

**生成的文件：**
- `Cleaned_Data/medal_counts_complete_mapping.csv`
- `Cleaned_Data/country_to_noc_mapping.json`
- `Cleaned_Data/checkpoint_medal_counts_after_step5.csv` (更新)

**关键验证点：**
- [ ] 映射成功率 > 90%
- [ ] 共同NOC > 130（目标：143）
- [ ] USA, CHN, GBR等关键国家都成功映射

---

#### **步骤5: 数据清洗 Part 2（2-3分钟）⭐ 最重要**

```bash
python Code/03_data_cleaning_part2_FIXED.py
```

**这个脚本做什么：**
- Step 6: 处理Programs表，计算每年的Total_Events
- Step 7: 创建东道主特征
  - is_host: 是否为东道主
  - years_since_hosted: 距离上次做东道主的年数
- Step 8: 创建时间特征
  - years_since_last: 距离上届奥运会的年数
  - crossed_war: 是否跨越战争年份
  - olympic_experience: 奥运参赛经验
- Step 9: 合并所有数据
- Step 10: 计算市场份额
  - Market_Share_Gold = Gold / Total_Events
  - Market_Share_Total = Total / Total_Events
- Step 11: 保存最终数据

**预期输出：**
```
✓ Total Events已计算: 29个年份
✓ 主办年份到NOC的映射: 33个记录，全部成功
✓ is_host特征已创建
  历史上做过东道主的次数: 90+

东道主效应验证:
  东道主平均金牌数: 23.27
  非东道主平均金牌数: 3.32
  东道主增幅: +19.95 枚 (+600.8%)
  ✓ 东道主效应验证成功！

✓ 市场份额已计算
✓✓✓ 数据清洗与预处理完成！
```

**生成的文件（最重要）：**
- `Cleaned_Data/master_data.csv` ⭐⭐⭐ 主数据文件
- `Cleaned_Data/total_events_by_year.csv`
- `Cleaned_Data/hosts_with_noc.csv`
- `Results/08_cleaning_report.csv`
- `Results/09_data_dictionary.csv`
- `Results/10_key_statistics.csv`

**关键验证点：**
- [ ] master_data.csv 存在且大小 > 100KB
- [ ] 东道主记录数 > 85（不是26！）
- [ ] 东道主增幅在 +200% 到 +700% 之间
- [ ] 市场份额已计算（非零值）

---

## ✅ 第三步：最终验证

### **快速Python验证**

```python
import pandas as pd

# 加载主数据
df = pd.read_csv('Cleaned_Data/master_data.csv')

print("="*60)
print("最终数据验证")
print("="*60)

# 基本信息
print(f"\n总记录数: {len(df):,}")
print(f"唯一NOC: {df['NOC'].nunique()}")
print(f"年份范围: {df['Year'].min()} - {df['Year'].max()}")
print(f"列数: {len(df.columns)}")

# 关键列检查
required = ['Year', 'NOC', 'Gold', 'is_host', 'Total_Events', 'Market_Share_Gold']
print(f"\n关键列存在: {all(col in df.columns for col in required)}")

# 东道主统计
print(f"\n东道主记录数: {df['is_host'].sum()}")
host_avg = df[df['is_host']==1]['Gold'].mean()
non_host_avg = df[df['is_host']==0]['Gold'].mean()
print(f"东道主平均金牌: {host_avg:.2f}")
print(f"非东道主平均金牌: {non_host_avg:.2f}")
boost_pct = ((host_avg / non_host_avg) - 1) * 100
print(f"东道主增幅: +{boost_pct:.0f}%")

# 2024年数据
df_2024 = df[df['Year'] == 2024]
print(f"\n2024年:")
print(f"  参赛国: {len(df_2024)}")
print(f"  总金牌: {df_2024['Gold'].sum()}")

# 前10名
print("\n2024年金牌榜前10:")
print(df_2024.nlargest(10, 'Gold')[['NOC', 'Gold', 'Total', 'is_host']])

print("\n" + "="*60)
if (len(df) > 1400 and 
    df['NOC'].nunique() > 140 and 
    df['is_host'].sum() > 85 and
    all(col in df.columns for col in required)):
    print("✓✓✓ 验证通过！数据质量优秀！")
else:
    print("⚠️ 部分检查未通过，请检查")
print("="*60)
```

---

## 📊 第四步：查看成果

### **生成的核心文件**

**Cleaned_Data/ (清洗后的数据)**
```
✓ master_data.csv           ⭐⭐⭐ 最重要！用于建模
  - 1,400+ 行
  - 19+ 列特征
  - 包含奖牌数、东道主特征、市场份额、时间特征
  
✓ cleaned_athletes.csv      去重后的运动员数据
✓ total_events_by_year.csv  每年总金牌数
```

**Results/ (分析报告)**
```
✓ 01_dataset_summary.csv       EDA摘要
✓ 02_data_quality_report.csv   质量报告
✓ 03-07_*.csv                  各类分析
✓ 08_cleaning_report.csv       清洗报告
✓ 09_data_dictionary.csv       数据字典
✓ 10_key_statistics.csv        关键统计
```

---

## 🎯 成功标志

### **完成后应该看到：**

✅ **文件检查**
- [ ] Cleaned_Data/ 有3个主要CSV
- [ ] Results/ 有10个报告文件
- [ ] master_data.csv > 100KB

✅ **数据质量**
- [ ] 记录数 > 1,400
- [ ] NOC数 > 140
- [ ] 东道主记录 > 85
- [ ] 市场份额已计算

✅ **关键指标**
- [ ] 2024年金牌数 ≈ 329
- [ ] 东道主增幅 +200%~700%
- [ ] 共同NOC = 143

---

## 🚀 完成后的下一步

### **1. 探索数据**
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Cleaned_Data/master_data.csv')

# 美国历年金牌趋势
usa = df[df['NOC'] == 'USA'].sort_values('Year')
plt.plot(usa['Year'], usa['Gold'])
plt.title('USA Gold Medals Over Time')
plt.show()

# 东道主效应可视化
import seaborn as sns
sns.boxplot(data=df, x='is_host', y='Gold')
plt.title('Host Effect on Gold Medals')
plt.show()
```

### **2. 特征工程**
- 创建滞后特征 (Lag features)
- 创建滚动统计 (Rolling statistics)
- 添加外部数据 (GDP, Population)

### **3. 建模**
- 时间序列模型 (ARIMA, Prophet)
- 机器学习模型 (Random Forest, XGBoost)
- 集成模型 (Stacking)

### **4. 预测2028**
- 预测各国金牌数
- 美国东道主效应 (+20%~30%)
- 生成预测区间

---

## 🆘 常见问题

### **Q1: 某个脚本运行失败**
→ 检查上一步是否成功完成
→ 确保checkpoint文件存在
→ 查看错误信息

### **Q2: NOC映射失败**
→ 确保Part 1成功运行
→ 重新运行create_complete_mapping.py

### **Q3: 东道主记录数只有26**
→ 说明NOC映射有问题
→ 重新运行create_complete_mapping.py

### **Q4: 需要完全重新开始**
→ 运行 clean_start.py
→ 按顺序重新运行5个脚本

---

## 📝 运行顺序总结

```bash
# 0. 清理（如果需要）
python clean_start.py

# 1. EDA
python Code/01_initial_eda.py           # 2分钟
python Code/02_investigate_issues.py    # 2分钟

# 2. 数据清洗
python Code/03_data_cleaning_part1.py   # 1分钟
python Code/create_complete_mapping.py  # 1分钟
python Code/03_data_cleaning_part2_FIXED.py  # 2分钟

# 3. 验证（Python代码）
# 运行上面的验证脚本

# 总时间: 8-10分钟
```

---

**准备好了吗？让我们开始！** 🚀

**第一步：**
```bash
python clean_start.py
```

然后告诉我结果！
