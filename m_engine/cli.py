#!/usr/bin/env python3
"""
M-Engine CLI: 记忆引擎命令行交互界面。

命令：
    ask <问题>              – 提问
    store <文本>            – 存储事实
    feedback <fact_id> <分数> – 对上次交互中的事实给予反馈（-1 到 1）
    show spectrum <fact_id>   – 查看事实频谱
    show basis               – 列出所有基函数
    show facts               – 列出所有事实
    show neural              – 显示最近一次交互的神经读数
    show dynamics            – 显示基函数动力学状态与统计
    show safety              – 显示第二序监控状态与警报
    explore                  – 探测认知缺口并生成探索性问题
    analogies                – 发现记忆间的结构类比
    export [fact_id]         – 导出事实为可交换JSON格式
    import <json_file>       – 从JSON文件导入事实
    evolve                   – 强制执行一次基函数演化
    save [路径]              – 保存状态到数据库
    load [路径]              – 从数据库加载状态
    exit                     – 退出
    help                     – 显示帮助

用法：
    python -m m_engine.cli
    或设置 OPENAI_API_KEY 后启动
"""

import logging
import os
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from m_engine.orchestrator import MEngineOrchestrator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("m_engine.cli")


def print_banner():
    print("=" * 56)
    print("  M-Engine  记忆计算原型引擎")
    print("  基于傅里叶类比的记忆压缩与重构系统")
    print("=" * 56)
    print("  命令: ask | store | feedback | show | save | load | help | exit")
    print("=" * 56)


def print_help():
    print("""
命令说明:
  ask <问题>                向记忆库提问
  store <文本>              存储一段文本作为事实
  feedback <fact_id> <分数> 对最近一次回答施加反馈（-1.0 ~ 1.0）
  show spectrum <fact_id>   查看事实在各基函数上的频谱
  show basis                列出所有基函数
  show facts                列出记忆库中的所有事实
  save [路径]               保存当前状态到数据库（默认 data/m_engine.db）
  load [路径]               从数据库加载状态
  exit                      退出
  help                      显示此帮助信息

使用流程示例:
  > store 小明因为考试不及格，躲在房间里哭了整整一个下午。
  > ask 小明为什么哭？
  > show spectrum fact_xxxxxxxx
  > feedback fact_xxxxxxxx 0.8
""")


def main():
    print_banner()

    # 获取数据目录
    package_dir = Path(__file__).resolve().parent
    data_dir = package_dir / "data"

    # 初始化引擎
    engine = MEngineOrchestrator(
        data_dir=str(data_dir),
        model=os.environ.get("MENGINE_MODEL", "gpt-4o-mini"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    try:
        engine.initialize()
    except FileNotFoundError as e:
        print(f"[错误] 数据文件未找到: {e}")
        print(f"请确认 {data_dir} 目录中存在 base_basis.json 和 base_questions.json")
        return
    except Exception as e:
        print(f"[错误] 初始化失败: {e}")
        return

    # API 后端检测优先级: DeepSeek > OpenAI > Ollama > mock
    ds_key = os.environ.get("DEEPSEEK_API_KEY", "")
    oa_key = os.environ.get("OPENAI_API_KEY", "")

    if ds_key:
        print("[提示] 使用 DeepSeek API (deepseek-chat)")
        engine.decoder.api_key = ds_key
        engine.decoder.base_url = "https://api.deepseek.com/v1"
        engine.decoder.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        engine.embedder.api_key = ds_key
        engine.embedder.base_url = "https://api.deepseek.com/v1"
        engine.embedder.model_name = None  # DeepSeek 无 embedding API，用 hash
    elif oa_key:
        print("[提示] 使用 OpenAI API")
    else:
        # 检测 Ollama
        ollama_available = False
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            ollama_available = True
        except Exception:
            pass
        if ollama_available:
            print("[提示] 检测到本地 Ollama，使用 gemma4:e4b + nomic-embed-text")
            engine.decoder.api_key = "ollama"
            engine.decoder.base_url = "http://localhost:11434/v1"
            engine.decoder.model = "gemma4:e4b"
            engine.embedder.base_url = "http://localhost:11434/v1"
            engine.embedder.model_name = "nomic-embed-text"
        else:
            print("[提示] 未检测到任何 API，使用模拟回答模式。")
            print("  设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 环境变量以启用真实 LLM。")
    print()

    # 加载预置示例事实
    _load_example_facts(engine)

    # 主循环
    while True:
        try:
            raw = input("M> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not raw:
            continue

        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "exit":
            print("再见。")
            break

        elif cmd == "help":
            print_help()

        elif cmd == "ask":
            if not args:
                print("[!] 用法: ask <问题>")
                continue
            print(f"\n[问题] {args}")
            print("-" * 40)
            answer, attention = engine.process_query(args)
            print(f"\n[回答]\n{answer}\n")
            if attention:
                print("[激活维度]")
                for name, strength in attention[:5]:
                    bar = "█" * max(1, int(strength * 20))
                    print(f"  [{bar}] {name}: {strength:.3f}")

        elif cmd == "store":
            if not args:
                print("[!] 用法: store <文本>")
                continue
            fact = engine.store_fact(args)
            print(f"[OK] 已存储事实: {fact.id}")
            print(f"   文本: {fact.raw_text[:80]}{'...' if len(fact.raw_text) > 80 else ''}")

        elif cmd == "feedback":
            fb_parts = args.split()
            if len(fb_parts) < 2:
                print("[!] 用法: feedback <fact_id> <分数>")
                print("   分数范围: -1.0 (完全否定) ~ 1.0 (完全肯定)")
                continue
            fact_id = fb_parts[0]
            try:
                score = float(fb_parts[1])
            except ValueError:
                print("[!] 分数必须是一个数字（-1.0 ~ 1.0）")
                continue
            result = engine.apply_feedback(fact_id, score)
            if result.get("status") == "ok":
                print(f"[OK] 反馈已应用: fact={fact_id} score={score}")
                if "changes" in result:
                    changed = result["changes"]
                    if changed:
                        print("   频谱变化:")
                        for bid, vals in list(changed.items())[:5]:
                            print(f"     {bid}: {vals['old']} → {vals['new']}")
            else:
                print(f"[FAIL] 反馈失败: {result.get('message', '未知错误')}")

        elif cmd == "analogies":
            analogies = engine.find_analogies()
            if not analogies:
                print("（未发现显著类比关系）")
            else:
                print(f"\n[类比] 发现 {len(analogies)} 对结构相似的记忆:")
                print("-" * 60)
                for i, a in enumerate(analogies, 1):
                    print(f"  {i}. 相似度={a['similarity']:.3f}")
                    print(f"     A: {a['fact_a']}...")
                    print(f"     B: {a['fact_b']}...")
                    print(f"     共享维度: {', '.join(a['dimensions'])}")
                    print(f"     {a['narrative']}")
                    print()

        elif cmd == "export":
            import json as _json
            fid = args if args else None
            data = engine.export_knowledge(fid)
            print(_json.dumps(data, ensure_ascii=False, indent=2))

        elif cmd == "import":
            if not args:
                print("[!] 用法: import <json_file>")
                continue
            import json as _json
            try:
                with open(args, "r", encoding="utf-8") as f:
                    data = _json.load(f)
                count = engine.import_knowledge(data if isinstance(data, list) else [data])
                print(f"[OK] 导入了 {count} 个事实")
            except FileNotFoundError:
                print(f"[FAIL] 文件不存在: {args}")
            except json.JSONDecodeError as e:
                print(f"[FAIL] JSON解析错误: {e}")

        elif cmd == "explore":
            count = 5
            if args:
                try:
                    count = int(args)
                except ValueError:
                    pass
            gaps = engine.explore_gaps(count)
            if not gaps:
                print("（未检测到认知缺口）")
            else:
                print(f"\n[探索] 认知缺口 (top {len(gaps)}):")
                print("-" * 60)
                for i, g in enumerate(gaps, 1):
                    if "message" in g:
                        print(f"  {g['message']}")
                    else:
                        print(f"  {i}. [{g['dimension']}] gap={g['gap']:.3f}")
                        print(f"     事实: {g['fact_preview']}...")
                        print(f"     问题: {g['question']}")
                        print()

        elif cmd == "evolve":
            events = engine.evolve_now()
            if not events:
                print("（暂无演化事件，可能在冷却中）")
            else:
                print(f"\n[演化] 产生了 {len(events)} 个事件:")
                for e in events:
                    print(f"  [{e['type']}] {e['reason']}")

        elif cmd == "save":
            path = args if args else None
            try:
                saved_path = engine.save(path)
                print(f"[OK] 状态已保存到: {saved_path}")
            except Exception as e:
                print(f"[FAIL] 保存失败: {e}")

        elif cmd == "load":
            path = args if args else None
            try:
                count = engine.load(path)
                print(f"[OK] 已从数据库加载 {count} 个事实")
            except Exception as e:
                print(f"[FAIL] 加载失败: {e}")

        elif cmd == "show":
            if not args:
                print("[!] 用法: show <spectrum|basis|facts> [参数]")
                continue
            show_parts = args.split(maxsplit=1)
            subcmd = show_parts[0].lower()
            subargs = show_parts[1] if len(show_parts) > 1 else ""

            if subcmd == "spectrum":
                if not subargs:
                    print("[!] 用法: show spectrum <fact_id>")
                    # 列出可用的事实 ID
                    facts = engine.fact_bus.list_all()
                    if facts:
                        print("可用事实 ID:")
                        for f in facts:
                            print(f"  {f.id}: {f.raw_text[:50]}...")
                    continue
                spec = engine.get_spectrum(subargs)
                if spec is None:
                    print(f"[FAIL] 未找到事实: {subargs}")
                else:
                    print(f"\n[频谱] 事实频谱: {spec['id']}")
                    print(f"   文本: {spec['text']}")
                    print("   ---")
                    # 按值排序
                    sorted_spec = sorted(spec['spectrum'].items(), key=lambda x: x[1], reverse=True)
                    for name, score in sorted_spec:
                        bar = "█" * max(1, int(score * 20))
                        print(f"  [{bar}] {name}: {score:.3f}")

            elif subcmd == "basis":
                basis_list = engine.basis_registry.list_all()
                print(f"\n[基函数] 基函数列表 ({len(basis_list)} 个):")
                print("-" * 40)
                for b in basis_list:
                    print(f"  [{b.id}] {b.name}")
                    print(f"    {b.description}")
                    print()

            elif subcmd == "facts":
                facts = engine.fact_bus.list_all()
                if not facts:
                    print("（记忆库为空）")
                else:
                    print(f"\n[事实] 事实列表 ({len(facts)} 个):")
                    print("-" * 40)
                    for f in facts:
                        print(f"  [{f.id}] {f.raw_text[:100]}{'...' if len(f.raw_text) > 100 else ''}")
                        print(f"   激活次数: {sum(f.activation.values())}")
                        print()

            elif subcmd == "safety":
                status = engine.get_safety_status()
                print(f"\n[安全] 第二序监控状态")
                for k, v in status.items():
                    print(f"  {k}: {v}")
                alerts = engine.get_safety_alerts(10)
                if alerts:
                    print("-" * 50)
                    print(f"最近警报 ({len(alerts)} 条):")
                    for a in alerts:
                        print(f"  [{a['type']}] {a.get('details', '')}")
                else:
                    print("  (无安全警报)")

            elif subcmd == "dynamics":
                status = engine.get_dynamics_status()
                if status.get("status") == "not_initialized":
                    print("（动力学未初始化）")
                else:
                    print(f"\n[动力学] 基函数演化状态")
                    print(f"  当前基函数数量: {status['total_basis']}")
                    print(f"  累计演化操作: {status['total_ops']}")
                    print(f"  冷却时间: {status['config']['cooldown_s']}s")
                    print(f"  每步上限: {status['config']['max_ops']}")
                    print("-" * 50)
                    print("基函数统计:")
                    for s in status.get("basis_stats", []):
                        bar = "█" * min(10, max(1, int(s['avg_strength'] * 50)))
                        print(f"  [{bar}] {s['name']}")
                        print(f"    激活次数={s['activation_count']} 均强={s['avg_strength']} "
                              f"方差={s['variance']} 代={s['generation']}")
                        print(f"    空闲={s['idle_seconds']:.0f}s")
                    if status.get("history"):
                        print("-" * 50)
                        print("最近演化事件:")
                        for h in status["history"]:
                            print(f"  [{h['type']}] {h['reason']}")

            elif subcmd == "neural":
                reading = engine.get_neural_reading()
                if reading is None:
                    print("（无神经读数，请先执行 ask 命令）")
                else:
                    print(f"\n[神经] 最近交互的双空间对比")
                    print(f"  logit_bias 注入 token 数: {reading['logit_bias_count']}")
                    print("-" * 50)
                    dims = reading["dimensions"]
                    # 按神经读数排序
                    sorted_dims = sorted(dims.items(), key=lambda x: abs(x[1]["neural"]), reverse=True)
                    for name, vals in sorted_dims:
                        s = vals["symbolic"]
                        n = vals["neural"]
                        d = vals["delta"]
                        s_bar = "█" * max(1, int(abs(s) * 20))
                        n_bar = "█" * max(1, int(abs(n) * 20))
                        sign = "+" if d >= 0 else ""
                        print(f"  {name}:")
                        print(f"    符号: [{s_bar}] {s:.4f}")
                        print(f"    神经: [{n_bar}] {n:.4f}")
                        print(f"    差值: {sign}{d:.4f}")
                    print()

            else:
                print(f"[!] 未知 show 子命令: {subcmd}，可用: spectrum, basis, facts")

        else:
            print(f"[!] 未知命令: {cmd}，输入 help 查看帮助")


def _load_example_facts(engine: MEngineOrchestrator):
    """加载预置示例事实，方便演示。"""
    examples = [
        "小明因为数学考试不及格，被爸爸严厉批评后，躲在房间里哭了整整一个下午。",
        "第二天，小明的好朋友小红来家里看望他，给他带来了一本数学练习册和自己做的笔记。",
        "小明的爸爸事后很后悔，他觉得不应该对儿子发那么大的火，但不知道该怎么开口道歉。",
        "一周后，小明在补考中取得了85分的好成绩，他第一时间告诉了小红和爸爸。",
    ]
    for text in examples:
        engine.store_fact(text)


if __name__ == "__main__":
    main()
