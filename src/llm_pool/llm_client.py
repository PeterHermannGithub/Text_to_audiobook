"""
LLM Client for distributed text processing.

This module provides a simple client interface for interacting with the LLM pool
from Spark workers and other distributed components.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from .llm_pool_manager import get_pool_manager, LLMPoolManager, LLMResponse


@dataclass
class LLMBatchRequest:
    """Represents a batch of LLM requests."""
    texts: List[str]
    model_config: Dict[str, Any]
    priority: int
    timeout: float
    batch_id: str


class LLMClient:
    """Client for interacting with LLM pool."""
    
    def __init__(self, pool_manager: LLMPoolManager = None):
        """Initialize LLM client."""
        self.pool_manager = pool_manager or get_pool_manager()
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    def classify_text(self, text: str, model_config: Dict[str, Any] = None, 
                     timeout: float = None) -> str:
        """Classify a single text using the LLM pool."""
        try:
            # Submit request
            request_id = self.pool_manager.submit_request(
                prompt=text,
                model_config=model_config,
                timeout=timeout
            )
            
            # Get response
            response = self.pool_manager.get_response(request_id, timeout)
            
            if response.success:
                return response.response_text
            else:
                raise Exception(f"LLM request failed: {response.error_message}")
                
        except Exception as e:
            self.logger.error(f"Error classifying text: {e}")
            raise
    
    def classify_texts_batch(self, texts: List[str], model_config: Dict[str, Any] = None,
                            timeout: float = None, max_workers: int = 5) -> List[str]:
        """Classify multiple texts in parallel."""
        if not texts:
            return []
        
        # Submit all requests
        request_ids = []
        for text in texts:
            request_id = self.pool_manager.submit_request(
                prompt=text,
                model_config=model_config,
                timeout=timeout
            )
            request_ids.append(request_id)
        
        # Collect responses
        results = []
        for request_id in request_ids:
            try:
                response = self.pool_manager.get_response(request_id, timeout)
                if response.success:
                    results.append(response.response_text)
                else:
                    results.append("")  # Empty result for failed requests
            except Exception as e:
                self.logger.error(f"Error getting response for {request_id}: {e}")
                results.append("")
        
        return results
    
    def classify_with_fallback(self, text: str, fallback_result: str = "AMBIGUOUS",
                              model_config: Dict[str, Any] = None, 
                              timeout: float = None) -> str:
        """Classify text with fallback on error."""
        try:
            return self.classify_text(text, model_config, timeout)
        except Exception as e:
            self.logger.warning(f"LLM classification failed, using fallback: {e}")
            return fallback_result
    
    def get_speaker_classifications(self, text_lines: List[str], 
                                   metadata: Dict[str, Any] = None,
                                   context_hint: str = None) -> List[str]:
        """Get speaker classifications for text lines."""
        from ..attribution.llm.prompt_factory import PromptFactory
        
        # Create prompt
        prompt_factory = PromptFactory()
        prompt = prompt_factory.create_speaker_classification_prompt(
            text_lines, metadata, context_hint
        )
        
        # Get classification
        try:
            response = self.classify_text(prompt)
            
            # Parse response to extract classifications
            classifications = self._parse_classification_response(response, len(text_lines))
            return classifications
            
        except Exception as e:
            self.logger.error(f"Error getting speaker classifications: {e}")
            return ["AMBIGUOUS"] * len(text_lines)
    
    def _parse_classification_response(self, response: str, expected_count: int) -> List[str]:
        """Parse classification response into list of speakers."""
        try:
            # Try to parse as JSON first
            import json
            parsed = json.loads(response)
            
            if isinstance(parsed, list) and len(parsed) == expected_count:
                return parsed
            elif isinstance(parsed, dict) and 'speakers' in parsed:
                speakers = parsed['speakers']
                if isinstance(speakers, list) and len(speakers) == expected_count:
                    return speakers
        except:
            pass
        
        # Fallback: try to extract from text
        lines = response.strip().split('\n')
        classifications = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                # Try to extract speaker name
                if ':' in line:
                    speaker = line.split(':')[0].strip()
                    classifications.append(speaker)
                else:
                    classifications.append(line)
        
        # Ensure we have the right number of classifications
        if len(classifications) != expected_count:
            self.logger.warning(f"Expected {expected_count} classifications, got {len(classifications)}")
            return ["AMBIGUOUS"] * expected_count
        
        return classifications
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of LLM pool."""
        return self.pool_manager.get_pool_status()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get LLM pool metrics."""
        return self.pool_manager.metrics_collector.get_metrics_summary()


class SparkLLMClient:
    """Specialized LLM client for Spark workers."""
    
    def __init__(self, worker_id: str = None):
        """Initialize Spark LLM client."""
        self.worker_id = worker_id or "spark-worker"
        self.client = LLMClient()
        self.logger = logging.getLogger(__name__)
    
    def process_chunk(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a chunk of text using LLM."""
        try:
            chunk_id = chunk_data['chunk_id']
            text_lines = chunk_data['text_lines']
            metadata = chunk_data.get('metadata', {})
            context_hint = chunk_data.get('context_hint')
            
            # Get classifications
            classifications = self.client.get_speaker_classifications(
                text_lines, metadata, context_hint
            )
            
            # Create result
            result = {
                'chunk_id': chunk_id,
                'worker_id': self.worker_id,
                'classifications': classifications,
                'processed_at': time.time(),
                'success': True
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing chunk {chunk_data.get('chunk_id')}: {e}")
            return {
                'chunk_id': chunk_data.get('chunk_id'),
                'worker_id': self.worker_id,
                'classifications': ["AMBIGUOUS"] * len(chunk_data.get('text_lines', [])),
                'processed_at': time.time(),
                'success': False,
                'error': str(e)
            }
    
    def process_chunks_batch(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple chunks in batch."""
        results = []
        
        for chunk in chunks:
            result = self.process_chunk(chunk)
            results.append(result)
        
        return results


class AsyncLLMClient:
    """Asynchronous LLM client for high-throughput scenarios."""
    
    def __init__(self, pool_manager: LLMPoolManager = None):
        """Initialize async LLM client."""
        self.pool_manager = pool_manager or get_pool_manager()
        self.logger = logging.getLogger(__name__)
    
    async def classify_text_async(self, text: str, model_config: Dict[str, Any] = None,
                                 timeout: float = None) -> str:
        """Classify text asynchronously."""
        loop = asyncio.get_event_loop()
        
        # Submit request
        request_id = await loop.run_in_executor(
            None, self.pool_manager.submit_request, text, model_config, 0, timeout
        )
        
        # Get response
        response = await loop.run_in_executor(
            None, self.pool_manager.get_response, request_id, timeout
        )
        
        if response.success:
            return response.response_text
        else:
            raise Exception(f"LLM request failed: {response.error_message}")
    
    async def classify_texts_batch_async(self, texts: List[str], 
                                        model_config: Dict[str, Any] = None,
                                        timeout: float = None) -> List[str]:
        """Classify multiple texts asynchronously."""
        if not texts:
            return []
        
        # Create tasks
        tasks = []
        for text in texts:
            task = self.classify_text_async(text, model_config, timeout)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Async classification failed: {result}")
                processed_results.append("")
            else:
                processed_results.append(result)
        
        return processed_results


class LLMClientPool:
    """Pool of LLM clients for high-concurrency scenarios."""
    
    def __init__(self, pool_size: int = 5):
        """Initialize LLM client pool."""
        self.pool_size = pool_size
        self.clients = [LLMClient() for _ in range(pool_size)]
        self.current_index = 0
        self.lock = asyncio.Lock()
    
    async def get_client(self) -> LLMClient:
        """Get an available client from the pool."""
        async with self.lock:
            client = self.clients[self.current_index]
            self.current_index = (self.current_index + 1) % self.pool_size
            return client
    
    async def classify_text_pooled(self, text: str, model_config: Dict[str, Any] = None,
                                  timeout: float = None) -> str:
        """Classify text using pooled clients."""
        client = await self.get_client()
        return client.classify_text(text, model_config, timeout)


# Utility functions for easy integration
def create_llm_client(config: Dict[str, Any] = None) -> LLMClient:
    """Create a new LLM client with optional configuration."""
    if config:
        from .llm_pool_manager import initialize_pool_manager
        pool_manager = initialize_pool_manager(config)
        return LLMClient(pool_manager)
    else:
        return LLMClient()


def create_spark_llm_client(worker_id: str = None) -> SparkLLMClient:
    """Create a new Spark LLM client."""
    return SparkLLMClient(worker_id)


def create_async_llm_client(config: Dict[str, Any] = None) -> AsyncLLMClient:
    """Create a new async LLM client."""
    if config:
        from .llm_pool_manager import initialize_pool_manager
        pool_manager = initialize_pool_manager(config)
        return AsyncLLMClient(pool_manager)
    else:
        return AsyncLLMClient()


# Context manager for LLM client
class LLMClientContext:
    """Context manager for LLM client lifecycle."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config
        self.client = None
    
    def __enter__(self) -> LLMClient:
        self.client = create_llm_client(self.config)
        return self.client
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client and self.client.pool_manager:
            self.client.pool_manager.stop()


# Decorator for automatic LLM client management
def with_llm_client(config: Dict[str, Any] = None):
    """Decorator for automatic LLM client management."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with LLMClientContext(config) as client:
                return func(client, *args, **kwargs)
        return wrapper
    return decorator