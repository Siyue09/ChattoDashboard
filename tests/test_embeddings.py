"""Remote embedding transport tests use synthetic responses only; no API calls."""
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from dashboard_agent import DashboardAgent
from dashboard_agent.data import write_assets
from dashboard_agent.providers import OpenAICompatibleEmbeddingProvider, ProviderError
from dashboard_agent.retrieval import SchemaRetriever


CONFIG = {"OPENAI_API_KEY": "unit-test-key", "OPENAI_BASE_URL": "https://example.invalid/v1",
          "OPENAI_EMBEDDING_MODEL": "unit-test-embedding", "RETRIEVAL_EMBEDDINGS": "remote"}


def response(data):
    return io.BytesIO(json.dumps({"data": data}).encode())


class EmbeddingProviderTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, CONFIG)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_configuration_is_explicit_and_does_not_make_a_request(self):
        for missing in ("OPENAI_API_KEY", "OPENAI_EMBEDDING_MODEL"):
            with self.subTest(missing=missing), patch.dict(os.environ, {missing: ""}):
                with patch("dashboard_agent.providers.urllib.request.urlopen") as network:
                    with self.assertRaises(ProviderError):
                        OpenAICompatibleEmbeddingProvider()
                network.assert_not_called()
        for endpoint in ("http://example.invalid/v1", "file:///tmp/endpoint", "https://username:password@example.invalid", "https://example.invalid?key=secret"):
            with self.subTest(endpoint=endpoint), patch.dict(os.environ, {"OPENAI_BASE_URL": endpoint}):
                with self.assertRaises(ProviderError):
                    OpenAICompatibleEmbeddingProvider()

    def test_batches_orders_and_normalizes_vectors(self):
        requests = []

        def transport(request, timeout):
            payload = json.loads(request.data)
            requests.append(payload)
            self.assertEqual(timeout, 30)
            self.assertEqual(request.full_url, "https://example.invalid/v1/embeddings")
            self.assertEqual(payload["encoding_format"], "float")
            return response([{"index": index, "embedding": [3.0, 4.0 + index]} for index in reversed(range(len(payload["input"])))])

        provider = OpenAICompatibleEmbeddingProvider(batch_size=2)
        with patch("dashboard_agent.providers.urllib.request.urlopen", side_effect=transport):
            vectors = provider.embed(["first document", "second document", "question"])
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["input"], ["first document", "second document"])
        self.assertEqual(vectors[0], (0.6, 0.8))
        self.assertAlmostEqual(vectors[1][0], 3 / math.sqrt(34))
        self.assertTrue(all(abs(math.hypot(*vector) - 1) < 1e-12 for vector in vectors))
        self.assertEqual(provider.dimension, 2)
        self.assertEqual(provider.request_count, 2)

    def test_rejects_invalid_vectors_and_indices(self):
        bad_data = [
            [{"index": 0, "embedding": [0, 0]}],
            [{"index": 0, "embedding": [float("nan"), 1]}],
            [{"index": 0, "embedding": [float("inf"), 1]}],
            [{"index": 0, "embedding": [True, 1]}],
            [{"index": 0, "embedding": ["1", "2"]}],
            [{"index": 0, "embedding": []}],
            [{"index": 1, "embedding": [1, 0]}],
            [{"index": True, "embedding": [1, 0]}],
            [],
        ]
        for data in bad_data:
            with self.subTest(data=data), patch("dashboard_agent.providers.urllib.request.urlopen", return_value=response(data)):
                with self.assertRaises(ProviderError):
                    OpenAICompatibleEmbeddingProvider().embed(["query"])
        duplicate = [{"index": 0, "embedding": [1, 0]}, {"index": 0, "embedding": [0, 1]}]
        with patch("dashboard_agent.providers.urllib.request.urlopen", return_value=response(duplicate)):
            with self.assertRaises(ProviderError):
                OpenAICompatibleEmbeddingProvider().embed(["one", "two"])

    def test_dimension_cannot_change_between_batches_or_questions(self):
        provider = OpenAICompatibleEmbeddingProvider(batch_size=1)
        replies = [response([{"index": 0, "embedding": [1, 0]}]), response([{"index": 0, "embedding": [1, 0, 0]}])]
        with patch("dashboard_agent.providers.urllib.request.urlopen", side_effect=replies):
            provider.embed(["document"])
            with self.assertRaisesRegex(ProviderError, "dimension changed"):
                provider.embed(["question"])

    def test_invalid_inputs_fail_before_network(self):
        provider = OpenAICompatibleEmbeddingProvider()
        for texts in ([""], [" "], ["a" * 16001], [None], "not a list", ["x"] * 2049):
            with self.subTest(type=type(texts)), patch("dashboard_agent.providers.urllib.request.urlopen") as network:
                with self.assertRaises(ProviderError):
                    provider.embed(texts)
                network.assert_not_called()
        self.assertEqual(provider.embed([]), [])

    def test_http_failures_are_safe_and_do_not_retry(self):
        failure = urllib.error.HTTPError("https://example.invalid", 401, "secret user prompt", {}, io.BytesIO(b"secret credential"))
        provider = OpenAICompatibleEmbeddingProvider()
        with patch("dashboard_agent.providers.urllib.request.urlopen", side_effect=failure) as network:
            with self.assertRaises(ProviderError) as caught:
                provider.embed(["user question"])
        self.assertEqual(str(caught.exception), "Embedding request failed with HTTP 401.")
        self.assertNotIn("secret", str(caught.exception))
        self.assertEqual(network.call_count, 1)


class EmbeddingRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.assets = write_assets(Path(self.temp.name) / "assets")

    def retriever(self, mode=None):
        return SchemaRetriever(self.assets["catalog"], self.assets["examples"], embeddings_mode=mode)

    def test_default_local_backend_remains_network_free(self):
        with patch.dict(os.environ, {"RETRIEVAL_EMBEDDINGS": "local"}):
            with patch("dashboard_agent.retrieval.OpenAICompatibleEmbeddingProvider") as provider:
                result = self.retriever().retrieve("Revenue by month")
        self.assertEqual(result["method"], "hashed_lexical_vectors_plus_schema_graph")
        self.assertEqual(result["embedding_backend"], "local")
        provider.assert_not_called()

    def test_semantic_backend_embeds_examples_and_shares_seed_scores(self):
        inputs = []

        def transport(request, timeout):
            batch = json.loads(request.data)["input"]
            inputs.extend(batch)
            # Synthetic feature vectors make ordering deterministic without
            # pretending this mock is a learned embedding model.
            return response([{"index": index, "embedding": [1.0, float(text.lower().count("revenue")), float(text.lower().count("month"))]}
                             for index, text in enumerate(batch)])

        with patch.dict(os.environ, CONFIG), patch("dashboard_agent.providers.urllib.request.urlopen", side_effect=transport) as network:
            retriever = self.retriever()
            vector = retriever.retrieve("Revenue by month", mode="vector")
            first_calls = network.call_count
            graph = retriever.retrieve("Revenue by month", mode="graph")
            self.assertEqual(network.call_count, first_calls + 1)
        self.assertEqual(first_calls, math.ceil((len(retriever.documents) + len(retriever.examples)) / 64) + 1)
        self.assertEqual(len(inputs), len(retriever.documents) + len(retriever.examples) + 2)
        self.assertTrue(all(example["question"] in inputs for example in retriever.examples))
        self.assertEqual(vector["evidence"], [entry for entry in graph["evidence"] if entry["reason"] == "semantic_vector_seed"])
        self.assertEqual(vector["context"]["examples"], graph["context"]["examples"])
        self.assertEqual(graph["method"], "remote_semantic_embeddings_plus_schema_graph")
        self.assertEqual(graph["embedding_model"], "unit-test-embedding")
        self.assertTrue(set(vector["tables"]) <= set(graph["tables"]))

    def test_remote_configuration_failure_is_visible_without_silent_fallback(self):
        with patch.dict(os.environ, {**CONFIG, "OPENAI_EMBEDDING_MODEL": ""}):
            with patch("dashboard_agent.providers.urllib.request.urlopen") as network:
                agent = DashboardAgent(str(Path(self.temp.name) / "unused.sqlite"), self.assets["catalog"], self.assets["examples"])
                result = agent.run("Revenue by month")
        self.assertEqual(result["status"], "error")
        self.assertIn("OPENAI_EMBEDDING_MODEL", result["answer"])
        self.assertEqual(result["rows"], [])
        self.assertNotIn("execute", [step["node"] for step in result["trace"]])
        network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
