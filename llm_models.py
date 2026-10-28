# -*- coding: utf-8 -*-
"""
双色球 LLM 大模型预测生成模块
通过 OpenAI 兼容代理调用 4 个大模型，每个模型输出 5 重策略，共 20 组预测。
仅依赖标准库 + requests（爬虫已依赖）。
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests

# ==================== 配置 ====================
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai-proxy.org/v1").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "60"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "800"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
LLM_HISTORY_WINDOW = int(os.environ.get("LLM_HISTORY_WINDOW", "30"))

MODELS = [
    {"model_id": "kimi-k2.5", "model_name": "Kimi K2.5"},
    {"model_id": "glm-5", "model_name": "GLM-5"},
    {"model_id": "qwen3.5-flash", "model_name": "Qwen3.5 Flash"},
    # deepseek-v4.1-flash 为推理模型：默认会把 max_tokens 全部用于思考(reasoning_content),
    # 导致 content 为空。用 reasoning_effort=none 关闭思考,直接输出且更省 token。
    {"model_id": "deepseek-v4.1-flash", "model_name": "DeepSeek V4.1 Flash",
     "extra_params": {"reasoning_effort": "none"}},
]

STRATEGIES = ["热号延续", "冷号回补", "区间均衡", "奇偶平衡", "和值回归"]

SYSTEM_PROMPT = (
    "你是双色球数据分析助手。基于给定历史开奖数据，输出5种不同策略的下期预测。"
    "严格只返回JSON数组，不要任何解释、markdown代码块或思考过程。"
)


# ==================== Prompt 构建 ====================

def _format_history(history_data: List[Dict[str, Any]]) -> str:
    """将最近 N 期历史压缩为极简文本，控制输入 token。"""
    lines = []
    for d in history_data[:LLM_HISTORY_WINDOW]:
        if not isinstance(d, dict):
            continue
        period = d.get("period", "?")
        reds = " ".join(str(b) for b in d.get("red_balls", []))
        blue = d.get("blue_ball", "?")
        lines.append(f"{period}:{reds}+{blue}")
    return "\n".join(lines)


def _build_user_prompt(history_text: str) -> str:
    strat_desc = "、".join(f"{i+1}.{s}" for i, s in enumerate(STRATEGIES))
    return (
        f"最近{LLM_HISTORY_WINDOW}期开奖(期号:红1..红6+蓝):\n{history_text}\n\n"
        f"请输出5种策略的预测，策略分别为:{strat_desc}。\n"
        "要求:红球6个(01-33,升序,2位零填充字符串),蓝球1个(01-16,2位零填充字符串)。\n"
        "5组红蓝组合不能完全相同,策略名用上面给定的名字。\n"
        "只返回此JSON(不要多余文字):\n"
        '[{"strategy":"热号延续","red_balls":["03","06",...],"blue_ball":"05","description":"一句话理由"},...]'
    )


# ==================== LLM 调用 ====================

def _call_llm(model_id: str, history_text: str,
              extra_params: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """调用单个 LLM，返回原始文本；失败返回 None。

    extra_params: 按模型注入的额外请求参数（如关闭推理模型的思考过程）。
    """
    if not LLM_API_KEY:
        print(f"  ⚠️  未配置 LLM_API_KEY，跳过 {model_id}")
        return None

    url = f"{LLM_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(history_text)},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }
    if extra_params:
        payload.update(extra_params)

    for attempt in range(2):  # 失败重试 1 次
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=LLM_TIMEOUT)
            if resp.status_code != 200:
                print(f"  ⚠️  {model_id} HTTP {resp.status_code}: {resp.text[:120]}")
                continue
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            print(f"  ⚠️  {model_id} 第{attempt+1}次调用异常: {type(e).__name__}: {e}")
    return None


# ==================== 解析 ====================

def _extract_json_array(text: str) -> Optional[list]:
    """从 LLM 输出中提取首个 JSON 数组（容错 markdown 围栏/前后杂文本）。"""
    if not text:
        return None
    # 去 markdown 代码围栏
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    # 找首个 [ 到匹配的 ]
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None


def _parse_groups(raw: Any, model_name: str) -> List[Dict[str, Any]]:
    """将 LLM 返回的数组规整为预测组列表，仅保留合法结构。"""
    if not isinstance(raw, list):
        return []
    groups = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        reds = item.get("red_balls")
        blue = item.get("blue_ball")
        if not isinstance(reds, list) or len(reds) != 6 or not blue:
            continue
        strategy = item.get("strategy") or STRATEGIES[i % len(STRATEGIES)]
        groups.append({
            "group_id": len(groups) + 1,
            "strategy": str(strategy),
            "red_balls": [str(b) for b in reds],
            "blue_ball": str(blue),
            "description": str(item.get("description", "")),
        })
    return groups


# ==================== 主入口 ====================

def _generate_one_model(model: Dict[str, Any], target_period: str,
                        prediction_date: str, history_text: str) -> Optional[Dict[str, Any]]:
    """生成单个 LLM 模型的预测，失败返回 None。"""
    mid = model["model_id"]
    mname = model["model_name"]
    content = _call_llm(mid, history_text, model.get("extra_params"))
    if content is None:
        print(f"  ✗ {mname}({mid}) 调用失败，跳过")
        return None

    raw = _extract_json_array(content)
    groups = _parse_groups(raw, mname)
    if not groups:
        print(f"  ✗ {mname}({mid}) 未解析出有效预测组，跳过")
        return None

    return {
        "prediction_date": prediction_date,
        "target_period": target_period,
        "model_type": "llm",
        "model_id": mid,
        "model_name": mname,
        "predictions": groups,
    }


def generate_llm_predictions(target_period: str, prediction_date: str,
                             history_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """并发调用 4 个 LLM，返回成功模型的预测列表（失败的已剔除）。"""
    history_text = _format_history(history_data)
    if not history_text:
        print("  ⚠️  历史数据为空，无法生成 LLM 预测")
        return []

    results = []
    with ThreadPoolExecutor(max_workers=len(MODELS)) as ex:
        futures = {
            ex.submit(_generate_one_model, m, target_period, prediction_date, history_text): m
            for m in MODELS
        }
        for fut in as_completed(futures):
            m = futures[fut]
            try:
                r = fut.result()
                if r:
                    print(f"  ✓ {m['model_name']}({m['model_id']}) 生成 {len(r['predictions'])} 组")
                    results.append(r)
            except Exception as e:
                print(f"  ✗ {m['model_name']}({m['model_id']}) 处理异常: {type(e).__name__}: {e}")

    # 按 MODELS 顺序稳定排序，保证输出可预期
    order = {m["model_id"]: i for i, m in enumerate(MODELS)}
    results.sort(key=lambda r: order.get(r["model_id"], 99))
    return results


if __name__ == "__main__":
    # 独立验证
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "data", "lottery_history.json"), "r", encoding="utf-8") as f:
        hist = json.load(f)
    preds = generate_llm_predictions("TEST", "2026-01-01", hist.get("data", []))
    print(json.dumps(preds, ensure_ascii=False, indent=2))
