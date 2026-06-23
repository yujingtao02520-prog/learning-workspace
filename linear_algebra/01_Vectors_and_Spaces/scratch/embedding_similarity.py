import numpy as np
import matplotlib.pyplot as plt

def cosine_similarity(v1, v2):
    """
    计算两个向量之间的余弦相似度
    公式: cos(theta) = (v1 . v2) / (||v1|| * ||v2||)
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # 防止除以零
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    return dot_product / (norm_v1 * norm_v2)

# ==========================================
# 1. 模拟语义空间：定义一些词的 Embedding (2维简化版)
# 假设 X 轴表示 "是否为食物" (0到1)，Y 轴表示 "是否有生命/会动" (0到1)
# ==========================================
embeddings = {
    "apple": np.array([0.9, 0.1]),   # 食物属性强，不会动
    "banana": np.array([0.95, 0.05]), # 食物属性强，不会动
    "cat": np.array([0.1, 0.95]),    # 活物属性强，不能吃
    "dog": np.array([0.05, 0.9]),    # 活物属性强，不能吃
    "pizza": np.array([0.85, 0.15]),  # 食物属性强，不会动
}

print("=== 向量相似度计算示例 ===")
# 计算 apple 和 banana 的相似度 (理论上应该很高，因为都是水果/食物)
sim_apple_banana = cosine_similarity(embeddings["apple"], embeddings["banana"])
print(f"apple 与 banana 的余弦相似度: {sim_apple_banana:.4f}")

# 计算 apple 和 cat 的相似度 (理论上应该很低)
sim_apple_cat = cosine_similarity(embeddings["apple"], embeddings["cat"])
print(f"apple 与 cat 的余弦相似度: {sim_apple_cat:.4f}")

# 计算 cat 和 dog 的相似度 (都是宠物，应该很高)
sim_cat_dog = cosine_similarity(embeddings["cat"], embeddings["dog"])
print(f"cat 与 dog 的余弦相似度: {sim_cat_dog:.4f}")


# ==========================================
# 2. 模拟微型语义检索系统 (Semantic Search)
# ==========================================
print("\n=== 模拟语义检索 ===")
query = np.array([0.8, 0.2]) # 这是一个偏向食物的输入向量 (例如用户想搜索 "水果" 相关的词)
print(f"检索 Query 向量: {query}")

scores = {}
for word, emb in embeddings.items():
    scores[word] = cosine_similarity(query, emb)

# 按相似度降序排序
sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
print("搜索排序结果:")
for word, score in sorted_scores:
    print(f" - {word}: 相似度 {score:.4f}")


# ==========================================
# 3. 几何可视化：在 2D 平面上画出这些向量
# ==========================================
plt.figure(figsize=(8, 8))
# 画坐标轴
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)

# 调色盘
colors = {'apple': 'red', 'banana': 'yellow', 'pizza': 'orange', 'cat': 'blue', 'dog': 'cyan'}

# 绘制数据库中的词向量
for word, emb in embeddings.items():
    plt.quiver(0, 0, emb[0], emb[1], angles='xy', scale_units='xy', scale=1, color=colors[word], label=word)
    plt.text(emb[0] + 0.02, emb[1] + 0.02, word, fontsize=12, fontweight='bold')

# 绘制查询词向量 (虚线箭头)
plt.quiver(0, 0, query[0], query[1], angles='xy', scale_units='xy', scale=1, color='purple', linestyle='--', label='Query (User Search)')
plt.text(query[0] + 0.02, query[1] - 0.04, 'Query', fontsize=12, color='purple', fontweight='bold')

plt.xlim(-0.1, 1.2)
plt.ylim(-0.1, 1.2)
plt.xlabel('Is Food? (食物属性)')
plt.ylabel('Is Alive? (活物属性)')
plt.title('Semantic Embedding Space & Vector Angle (2D)')
plt.grid(True, alpha=0.3)
plt.legend()
print("\n[提示] 绘图窗口已生成，请在支持 GUI 的终端下查看，或在本地运行以显示图表。")
plt.show()
