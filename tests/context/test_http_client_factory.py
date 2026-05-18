import unittest

from Context.http_client_factory import create_async_http_client, create_sync_http_client


class HttpClientFactoryTests(unittest.TestCase):
    def test_sync_http_client_disables_environment_proxy_inheritance(self):
        client = create_sync_http_client()
        try:
            self.assertFalse(client._trust_env)
        finally:
            client.close()

    def test_async_http_client_disables_environment_proxy_inheritance(self):
        client = create_async_http_client()
        try:
            self.assertFalse(client._trust_env)
        finally:
            import asyncio

            asyncio.run(client.aclose())


if __name__ == "__main__":
    unittest.main()
