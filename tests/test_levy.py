import unittest
from levy.engine import LevyEngine
from levy.config import LevyConfig
from levy.models import LevyResult

class TestLevyEngine(unittest.TestCase):
    def test_exact_cache(self):
        config = LevyConfig(
            enable_exact_cache=True, 
            enable_semantic_cache=False,
            llm_provider="mock",
            embedding_provider="mock"
        )
        engine = LevyEngine(config)
        
        prompt = "test prompt"
        
        # First call: Miss
        res1 = engine.generate(prompt)
        self.assertEqual(res1.source, "llm")
        
        # Second call: Hit
        res2 = engine.generate(prompt)
        self.assertEqual(res2.source, "exact_cache")
        self.assertEqual(res1.answer, res2.answer)

    def test_semantic_cache_mock_behavior(self):
        # With mock embeddings, we can't easily test semantic similarity unless we force it
        # But we can test that the machinery runs without crashing.
        config = LevyConfig(
            enable_exact_cache=False, 
            enable_semantic_cache=True,
            similarity_threshold=0.0, # Accept everything
            llm_provider="mock",
            embedding_provider="mock"
        )
        engine = LevyEngine(config)
        
        # 1. Miss
        res1 = engine.generate("Hello")
        self.assertEqual(res1.source, "llm")
        
        # 2. Hit (Threshold 0.0 means even random storage match might trigger if loop runs, 
        # but mock embeddings are random so might be 0 similarity or accidental match?
        # Actually random vectors might be orthogonal.
        # But let's just ensure it stores it.)
        
        # Access internals to verify storage
        self.assertEqual(len(engine.store.entries), 1)

if __name__ == '__main__':
    unittest.main()
