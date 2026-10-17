"""多轮对话，或使用固定题目比较基础模型和 LoRA。"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATION = {
    "max_new_tokens": 1024, "do_sample": True,
    "temperature": 0.7, "top_p": 0.8, "top_k": 20,
}


def generate_answer(model, tokenizer, messages, seed=None):
    import torch

    if seed is not None:
        torch.manual_seed(seed)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, **GENERATION, pad_token_id=tokenizer.eos_token_id)
    tokens = output[0, inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(tokens, skip_special_tokens=True).split("</think>")[-1].strip()
    eos_ids = model.generation_config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    return {
        "answer": answer,
        "truncated": len(tokens) == GENERATION["max_new_tokens"] and tokens[-1].item() not in eos_ids,
    }


def compare_models(model, tokenizer, system, cases, seed):
    results = []
    for number, case in enumerate(cases, start=1):
        # 每题使用预先写好的历史；两边都不沿用上一次生成的回答。
        messages = [
            {"role": "system", "content": system}, *case["history"],
            {"role": "user", "content": case["prompt"]},
        ]
        case_seed = seed + number - 1
        print(f"[{number}/{len(cases)}] 基础模型：{case['prompt'].splitlines()[0]}", flush=True)
        with model.disable_adapter():
            base = generate_answer(model, tokenizer, messages, case_seed)
        print(f"[{number}/{len(cases)}] LoRA", flush=True)
        lora = generate_answer(model, tokenizer, messages, case_seed)
        results.append({**case, "seed": case_seed, "base": base, "lora": lora, "verdict": None})
    return results


def write_comparison(results, system, seed):
    folder = ROOT / "outputs/code-tutor"
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"comparison_seed{seed}"
    record = {"system": system, "enable_thinking": False, "generation": GENERATION, "results": results}
    stem.with_suffix(".json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# 基础模型与 LoRA 对照", "",
        "两边使用相同的系统提示、固定历史、采样参数和逐题种子。结论需要人工填写。", "",
        "先检查技术正确性，再检查教学帮助和给代码的时机。标记截断的回答需要补测。",
    ]
    for number, result in enumerate(results, start=1):
        lines.extend(["", f"## 第 {number} 题（seed={result['seed']}）", "", "### 固定历史", ""])
        for message in result["history"]:
            role = "学生" if message["role"] == "user" else "助教"
            lines.extend([f"**{role}：**", "", message["content"], ""])
        if not result["history"]:
            lines.append("无。")
        lines.extend(["", "### 当前提问", "", result["prompt"], "", "### 检查要点", ""])
        lines.extend(f"- {item}" for item in result["rubric"])
        for key, label in (("base", "基础模型"), ("lora", "LoRA")):
            lines.extend(["", f"### {label}", ""])
            if result[key]["truncated"]:
                lines.extend(["**达到生成长度上限，回答可能不完整，需要补测。**", ""])
            lines.append(result[key]["answer"])
        lines.extend(["", "### 人工结论", "", "待填写：LoRA 更好 / 基础模型更好 / 相当 / 均未达标。理由：", ""])
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"对照结果：{stem.with_suffix('.json')}\n阅读并评分：{stem.with_suffix('.md')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--base", action="store_true", help="只用基础模型对话")
    mode.add_argument("--compare", action="store_true", help="自动运行固定题目，保存基础模型与 LoRA 回答")
    parser.add_argument("--seed", type=int, default=42, help="对照模式的起始种子，默认 42")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = str(ROOT / "models/Qwen3-8B")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa",
    )
    if not args.base:
        model = PeftModel.from_pretrained(model, str(ROOT / "outputs/code-tutor/lora"))
    model.eval()

    system = (ROOT / "prompts/code_tutor_system.txt").read_text(encoding="utf-8").strip()
    if args.compare:
        cases = json.loads((ROOT / "data/code_tutor_eval.json").read_text(encoding="utf-8"))
        results = compare_models(model, tokenizer, system, cases, args.seed)
        write_comparison(results, system, args.seed)
        return

    messages = [{"role": "system", "content": system}]
    print("输入问题开始对话；:clear 清空历史；:quit 退出。")
    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question == ":quit":
            break
        if question == ":clear":
            messages = messages[:1]
            print("已清空历史。")
            continue
        if not question:
            continue

        messages.append({"role": "user", "content": question})
        result = generate_answer(model, tokenizer, messages)
        answer = result["answer"]
        print(f"\n助教：{answer}")
        if result["truncated"]:
            print("（回答达到生成长度上限，可能不完整。）")
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
