"""检查精简后仍保留的数据分组和多轮训练约定，无需 GPU。"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_data import split_records, training_record


class PrepareDataTests(unittest.TestCase):
    def test_related_turns_stay_together(self):
        records = json.loads((ROOT / "data/code_tutor_seed.json").read_text(encoding="utf-8"))
        train, validation = split_records(records)
        train_groups = {row["metadata"]["group_id"] for row in train}
        validation_groups = {row["metadata"]["group_id"] for row in validation}
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertFalse(train_groups & validation_groups)
        self.assertEqual(len(train) + len(validation), len(records))
        self.assertEqual((train, validation), split_records(records))

    def test_training_keeps_history_and_uses_shared_prompt(self):
        rows = json.loads((ROOT / "data/code_tutor_seed.json").read_text(encoding="utf-8"))
        system = (ROOT / "prompts/code_tutor_system.txt").read_text(encoding="utf-8").strip()
        for row in rows:
            result = training_record(row, system)
            self.assertEqual(result["history"], row.get("history", []))
            self.assertEqual(result["output"], row["output"])
            self.assertEqual(result["system"], system)
            self.assertNotIn("metadata", result)

    def test_one_topic_cannot_form_two_sets(self):
        with self.assertRaises(ValueError):
            split_records([{"metadata": {"group_id": "same"}}] * 3)


if __name__ == "__main__":
    unittest.main()
