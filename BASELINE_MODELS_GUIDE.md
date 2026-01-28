# 建模脚本说明文档
# Baseline Models Documentation

## 🎯 脚本概述

**文件名:** `05_baseline_models.py`

**功能:** 训练并对比6个基线模型，包括论文中的TMH-OMP模型

**预计运行时间:** 5-10分钟

---

## 📊 包含的6个模型

### **1. 线性回归 (Linear Regression)**
- **类型:** 基础统计模型
- **优点:** 简单、快速、可解释
- **缺点:** 假设线性关系
- **适用场景:** 基线参考

### **2. 泊松回归 (Poisson Regression)**
- **类型:** 广义线性模型
- **优点:** 专门处理计数数据
- **缺点:** 假设方差=均值
- **适用场景:** 奖牌数（非负整数）

### **3. Random Forest**
- **类型:** 集成树模型
- **优点:** 稳健、处理非线性、特征重要性
- **缺点:** 可解释性较低
- **适用场景:** 复杂关系建模

### **4. XGBoost**
- **类型:** 梯度提升树
- **优点:** 高精度、快速、处理缺失值
- **缺点:** 需要调参
- **适用场景:** 竞赛级预测

### **5. TMH-OMP ⭐ 论文模型**
- **类型:** Tobit回归 + Mundlak修正
- **优点:** 
  - 处理截断数据（奖牌≥0）
  - 国家固定效应
  - 时间不变因素
- **缺点:** 实现复杂
- **适用场景:** 学术研究、论文复现

**TMH-OMP实现细节:**
- **Tobit效应:** 截断负预测值为0
- **Mundlak修正:** 添加时间平均特征
  - Gold_mean, Total_mean, is_host_mean等
- **固定效应:** NOC_encoded（国家编码）
- **关键变量:**
  - HE (Host Effect): is_host
  - ER (Event Rate): Total_Events
  - AP (Athlete Performance): Gold_lag1等
  - NA (Number of Athletes): Num_Athletes

### **6. 集成模型 (Ensemble)**
- **类型:** RF + XGBoost平均
- **优点:** 结合多个模型优势
- **缺点:** 增加复杂度
- **适用场景:** 追求最高精度

---

## 🔧 脚本功能模块

### **Part 1: 数据加载**
- 加载特征工程后的数据
- 加载运动员数据（计算NA）

### **Part 2: TMH-OMP变量准备**
- ✅ 计算每国每年运动员数 (NA)
- ✅ 计算Mundlak时间平均值
- ✅ 创建国家固定效应编码

### **Part 3: 数据分割**
- **训练集:** 1896-2016 (约1,200条)
- **验证集:** 2020 (约91条)
- **测试集:** 2024 (约91条)
- 缺失值填充为0

### **Part 4: 模型训练**
- 按顺序训练6个模型
- 自动处理异常
- 保存所有预测结果

### **Part 5: 模型评估**
- **指标:**
  - RMSE (均方根误差) - 主要指标
  - MAE (平均绝对误差)
  - R² (决定系数)
  - MAPE (平均绝对百分比误差)
- 对比所有模型
- 选择最佳模型

### **Part 6: 可视化**
生成4张图表：
1. **模型RMSE对比** - 横向条形图
2. **实际vs预测** - 散点图
3. **前10名国家对比** - 柱状图
4. **特征重要性** - 横向条形图

### **Part 7: 特征重要性**
- Random Forest特征重要性
- 识别最重要的预测因子
- 保存CSV和图表

### **Part 8: 保存最佳模型**
- 保存模型文件 (.pkl)
- 保存特征列表 (.txt)
- 用于2028年预测

### **Part 9: 总结**
- 显示所有结果
- 指引下一步

---

## 📁 输出文件

### **Results/ (2个CSV)**
```
13_model_comparison.csv     模型性能对比
14_feature_importance.csv   特征重要性排名
```

### **Figures/ (4个PNG)**
```
06_model_rmse_comparison.png      RMSE对比图
07_actual_vs_predicted_best.png   预测效果图
08_top10_comparison.png           前10名对比
09_feature_importance.png         特征重要性图
```

### **Models/ (2个文件)**
```
best_model.pkl                    最佳模型（用于预测）
feature_list.txt                  特征列表
```

---

## 🎯 预期结果

### **模型性能（预期范围）**

基于历史奥运数据的特点：

| 模型 | RMSE | MAE | R² |
|------|------|-----|-----|
| Linear Regression | 8-12 | 5-8 | 0.6-0.7 |
| Poisson Regression | 8-11 | 5-7 | 0.65-0.75 |
| Random Forest | 6-9 | 4-6 | 0.75-0.85 |
| XGBoost | 5-8 | 3-5 | 0.8-0.9 |
| TMH-OMP | 7-10 | 4-7 | 0.7-0.8 |
| Ensemble | 5-7 | 3-5 | 0.82-0.92 |

**注意:** 实际结果取决于数据质量和特征质量

---

## 🔍 关键特征（预期TOP 10）

根据奥运预测的领域知识：

1. **Gold_lag1** ⭐⭐⭐ - 上届金牌数（最强！）
2. **Gold_rolling_mean_3** ⭐⭐⭐ - 近期平均
3. **is_host** ⭐⭐⭐ - 东道主效应
4. **Gold_host_boost** ⭐⭐ - 东道主加成
5. **Total_Events** ⭐⭐ - 比赛规模
6. **Num_Athletes** ⭐⭐ - 运动员数量
7. **Gold_growth_rate** ⭐ - 增长趋势
8. **Market_Share_Gold** ⭐ - 市场份额
9. **gold_percentile** ⭐ - 相对排名
10. **olympic_experience** ⭐ - 参赛经验

---

## ⚠️ 注意事项

### **1. TMH-OMP实现**
- 当前版本是**简化版**
- 使用Ridge回归模拟Tobit
- 包含Mundlak修正和固定效应
- **如果需要完整版**（真正的Tobit回归），需要使用`statsmodels`库

### **2. 缺失值处理**
- 滞后特征的缺失是正常的
- 已用0填充（表示无历史）
- 树模型可以处理NaN
- 如果需要，可以尝试其他填充策略

### **3. 模型选择**
- 脚本会自动选择RMSE最低的模型
- 但也要考虑：
  - 可解释性（论文需要）
  - 稳定性（多次运行结果）
  - 实用性（预测置信度）

### **4. 时间**
- 第一次运行可能需要5-10分钟
- Random Forest和XGBoost较慢
- 如果时间紧张，可以减少n_estimators

---

## 🚀 运行方法

### **方法1: 直接运行（推荐）**

```bash
# 确保脚本在Code文件夹
move 05_baseline_models.py Code/

# 运行
python Code/05_baseline_models.py
```

### **方法2: 在Python中运行**

```python
exec(open('Code/05_baseline_models.py').read())
```

---

## 📊 如何解读结果

### **看Results/13_model_comparison.csv:**

```csv
Model,Dataset,RMSE,MAE,R2,MAPE
Linear Regression,Validation (2020),8.5,5.2,0.68,45.3
Linear Regression,Test (2024),9.1,5.8,0.65,48.7
...
```

**关键指标:**
- **RMSE < 8:** 优秀
- **RMSE 8-10:** 良好
- **RMSE > 10:** 需要改进

- **R² > 0.8:** 优秀
- **R² 0.7-0.8:** 良好
- **R² < 0.7:** 需要改进

### **看图表:**
1. **RMSE对比图:** 哪个模型最好？
2. **预测散点图:** 预测准确吗？（点靠近红线=好）
3. **前10名对比:** 强队预测准确吗？
4. **特征重要性:** 哪些特征最重要？

---

## 🎯 下一步

### **如果模型效果好（R² > 0.75）:**
✅ 可以直接用于2028预测
→ 运行预测脚本

### **如果模型效果一般（R² 0.6-0.75）:**
🔧 可以改进：
- 添加更多特征
- 调整模型参数
- 尝试其他模型

### **如果TMH-OMP表现最好:**
📚 在论文中重点解释这个模型
→ 引用原论文
→ 解释Tobit和Mundlak

---

## 💡 常见问题

**Q1: TMH-OMP效果不如XGBoost？**
→ 正常！TMH-OMP是统计模型，XGBoost是机器学习
→ TMH-OMP优势在于可解释性和理论基础
→ 可以在论文中同时展示两者

**Q2: 所有模型R²都不高？**
→ 奥运预测本身就很难（政治、经济、体育政策等复杂因素）
→ R² > 0.7已经算不错
→ 关注相对排名而非绝对值

**Q3: 运行时间太长？**
→ 减少Random Forest的n_estimators（100→50）
→ 减少XGBoost的n_estimators（100→50）
→ 或者注释掉某些不需要的模型

**Q4: 特征重要性和预期不符？**
→ 可能是数据特性导致
→ 查看相关性矩阵
→ 尝试特征选择

---

## 📚 参考

**TMH-OMP模型来源:**
- 论文模型：Tobit-Mundlak-Hurdle Olympic Medal Prediction
- 核心思想：处理截断数据 + 固定效应 + 首次获奖预测

**机器学习模型:**
- scikit-learn文档
- XGBoost文档

---

**准备好了吗？运行脚本，让我们看看哪个模型最好！** 🚀
