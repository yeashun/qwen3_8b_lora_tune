"""无需模型，检查两边输入一致、LoRA 开关正确以及报告保留固定历史。"""
from contextlib import contextmanager
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import chat_tutor


class FakeAdapter:
    enabled = True

    @contextmanager
    def disable_adapter(self):
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = True


class ComparisonTests(unittest.TestCase):
    def test_identical_context_seed_and_correct_adapter_state_for_every_case(self):
        cases = json.loads((ROOT / "data/code_tutor_eval.json").read_text(encoding="utf-8"))
        original = copy.deepcopy(cases)
        model, calls = FakeAdapter(), []

        def generate(model, tokenizer, messages, seed):
            calls.append((model.enabled, copy.deepcopy(messages), seed))
            return {"answer": "LoRA" if model.enabled else "基础模型", "truncated": False}

        with patch.object(chat_tutor, "generate_answer", side_effect=generate), patch("sys.stdout", new_callable=io.StringIO):
            results = chat_tutor.compare_models(model, None, "统一提示", cases, 42)
        for i, case in enumerate(cases):
            base, lora = calls[2 * i:2 * i + 2]
            self.assertFalse(base[0])
            self.assertTrue(lora[0])
            self.assertEqual(base[1:], lora[1:])
            self.assertEqual(base[1][1:-1], case["history"])
            self.assertEqual(base[1][-1]["content"], case["prompt"])
            self.assertEqual(base[2], 42 + i)
            self.assertIsNone(results[i]["verdict"])
        self.assertEqual(cases, original)
        self.assertTrue(model.enabled)

    def test_reports_keep_history_and_flag_truncated_answers(self):
        result = {
            "prompt": "当前提问", "history": [{"role": "user", "content": "已有尝试"}],
            "rubric": ["检查原因"], "seed": 42, "verdict": None,
            "base": {"answer": "基础回答", "truncated": True},
            "lora": {"answer": "微调回答", "truncated": False},
        }
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(chat_tutor, "ROOT", Path(directory)), patch("sys.stdout", new_callable=io.StringIO):
                chat_tutor.write_comparison([result], "统一提示", 42)
            folder = Path(directory) / "outputs/code-tutor"
            record = json.loads((folder / "comparison_seed42.json").read_text(encoding="utf-8"))
            report = (folder / "comparison_seed42.md").read_text(encoding="utf-8")
            self.assertEqual(record["results"], [result])
            self.assertEqual(record["system"], "统一提示")
            for text in ("已有尝试", "当前提问", "检查原因", "基础回答", "微调回答", "可能不完整", "人工结论"):
                self.assertIn(text, report)


if __name__ == "__main__":
    unittest.main()
