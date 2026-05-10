# M-Engine 记忆计算原型引擎

基于傅里叶类比的记忆压缩与重构系统。

## 核心思想

记忆 = 有损压缩的事实 + 能还原事实骨架逻辑的"好问题"钩子

- 事实被压缩为在"基函数"（世界基本逻辑维度）上的投影系数（频谱）
- 不同问题/偏好从同一频谱中提取不同侧面的骨架，还原出不同的回答
- 通过用户反馈不断调整频谱，形成自适应记忆

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 设置 API Key（可选，不设置则使用模拟模式）
export OPENAI_API_KEY=your-key-here

# 启动 CLI
python -m m_engine.cli
```

## CLI 命令

| 命令 | 说明 |
|------|------|
| `ask <问题>` | 向记忆库提问 |
| `store <文本>` | 存储一段文本作为事实 |
| `feedback <fact_id> <分数>` | 对最近一次回答施加反馈 (-1.0~1.0) |
| `show spectrum <fact_id>` | 查看事实的频谱 |
| `show basis` | 列出所有基函数 |
| `show facts` | 列出所有事实 |
| `help` | 显示帮助 |

## 运行测试

```bash
pip install pytest
python -m pytest m_engine/tests/ -v
```

## 项目结构

```
m_engine/
├── __init__.py
├── cli.py                 # 命令行界面
├── orchestrator.py        # 主控协调器
├── core/
│   ├── basis_registry.py  # 基函数注册表
│   ├── fact_bus.py        # 事实总线（内存DB）
│   ├── question_router.py # 问题路由器
│   ├── preference_modem.py# 偏好调制器
│   ├── m_algebra.py       # M-代数核心
│   ├── decoder.py         # LLM 解码器
│   └── meta_updater.py    # 元更新网络
├── data/
│   ├── base_basis.json    # 10个基函数定义
│   └── base_questions.json# 10个基问题模板
└── tests/
    ├── test_core.py        # 单元测试
    └── test_integration.py # 集成测试
```
