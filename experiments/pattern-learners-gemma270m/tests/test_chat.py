from __future__ import annotations

import unittest

from chat import GenerationConfig, clean_generated_answer, parse_chat_command


class ChatTests(unittest.TestCase):
    def test_parse_chat_commands(self) -> None:
        self.assertEqual(parse_chat_command("/use addition"), ("use", "addition"))
        self.assertEqual(parse_chat_command("/quit"), ("exit", None))
        self.assertEqual(parse_chat_command("/none"), ("base", None))
        self.assertIsNone(parse_chat_command("What is 2 + 3?"))

    def test_clean_generated_answer(self) -> None:
        self.assertEqual(clean_generated_answer("  42\nQuestion: next"), "42")
        self.assertEqual(clean_generated_answer("\n7\n"), "7")
        self.assertEqual(clean_generated_answer("   "), "<empty response>")

    def test_generation_config_validation(self) -> None:
        GenerationConfig().validate()
        with self.assertRaises(ValueError):
            GenerationConfig(max_new_tokens=0).validate()
        with self.assertRaises(ValueError):
            GenerationConfig(temperature=-1).validate()


if __name__ == "__main__":
    unittest.main()
