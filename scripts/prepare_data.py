"""按题目分组划分数据，生成 LLaMA-Factory 所需的三个 JSON 文件。"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def split_records(records):
    # 同一道题的不同轮次必须放在一起，避免验证集提前见过训练题。
    groups = list(dict.fromkeys(row["metadata"]["group_id"] for row in records))
    if len(groups) < 2:
        raise ValueError("至少需要两个不同的 group_id 才能划分训练集和验证集。")
    random.Random(42).shuffle(groups)
    validation_groups = set(groups[:max(1, round(len(groups) * 0.2))])
    train = [row for row in records if row["metadata"]["group_id"] not in validation_groups]
    validation = [row for row in records if row["metadata"]["group_id"] in validation_groups]
    return train, validation


def training_record(row, system):
    # metadata 只用于分组，不交给模型学习。
    return {
        "instruction": row["instruction"],
        "input": row.get("input", ""),
        "output": row["output"],
        "system": system,
        "history": row.get("history", []),
    }


def main():
    records = json.loads((ROOT / "data/code_tutor_seed.json").read_text(encoding="utf-8"))
    system = (ROOT / "prompts/code_tutor_system.txt").read_text(encoding="utf-8").strip()
    train, validation = split_records(records)
    output_dir = ROOT / "data/processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = {}
    for name, rows in (("train", train), ("validation", validation)):
        data = [training_record(row, system) for row in rows]
        (output_dir / f"{name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        registry[f"code_tutor_{name}"] = {
            "file_name": f"{name}.json",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction", "query": "input", "response": "output",
                "system": "system", "history": "history",
            },
        }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"已生成 {len(train)} 条训练数据、{len(validation)} 条验证数据：{output_dir}")


if __name__ == "__main__":
    main()
