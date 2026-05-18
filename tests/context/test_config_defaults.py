import importlib
import os
import tempfile
import unittest
from unittest.mock import patch


class ConfigDefaultsTests(unittest.TestCase):
    def test_default_chat_model_uses_minimax(self):
        with patch.dict(os.environ, {}, clear=True):
            import Context.config as config

            reloaded = importlib.reload(config)
            self.assertEqual(reloaded.OPENAI_MODEL, "MiniMax-M2.7")
            self.assertEqual(reloaded.OPENAI_BASE_URL, "https://api.minimaxi.com/v1")

    def test_config_can_load_from_local_env_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False, encoding="utf-8") as handle:
            handle.write("OPENAI_API_KEY=test-key\n")
            handle.write("OPENAI_BASE_URL=https://api.minimaxi.com/v1\n")
            handle.write("OPENAI_MODEL=MiniMax-M2.7\n")
            env_path = handle.name

        try:
            with patch.dict(os.environ, {"AGENT_SYSTEM_ENV_FILE": env_path}, clear=True):
                import Context.config as config

                reloaded = importlib.reload(config)
                self.assertEqual(reloaded.OPENAI_API_KEY, "test-key")
                self.assertEqual(reloaded.OPENAI_BASE_URL, "https://api.minimaxi.com/v1")
                self.assertEqual(reloaded.OPENAI_MODEL, "MiniMax-M2.7")
        finally:
            os.unlink(env_path)


if __name__ == "__main__":
    unittest.main()
