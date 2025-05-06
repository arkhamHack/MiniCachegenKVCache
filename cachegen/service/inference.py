import asyncio
from typing import AsyncGenerator, Dict, List, Optional
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from .metrics import metrics, router as metrics_router

from ..kv_cache.pipeline import KVCachePipeline, PipelineConfig
from ..kv_cache.chunking import ChunkConfig, KVChunkManager
from ..kv_cache.exceptions import KVCacheError

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = True

class LLMService:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        device: str = "cuda",
        chunk_config: Optional[ChunkConfig] = None,
        pipeline_config: Optional[PipelineConfig] = None
    ):
        """Initialize LLM service with CacheGen integration.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run model on
            chunk_config: KV cache chunking configuration
            pipeline_config: Pipeline configuration for async operations
        """
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize CacheGen components
        self.chunk_config = chunk_config or ChunkConfig()
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.chunk_manager = KVChunkManager(self.chunk_config)
        self.pipeline = KVCachePipeline(self.pipeline_config)
        
        # Track active generations
        self._active_generations: Dict[str, Dict] = {}
    
    async def _stream_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gen_kwargs: Dict,
        use_cachegen: bool
    ) -> AsyncGenerator[str, None]:
        """Generate text with KV cache management."""
        start_time = time.time()
        try:
            # Initial forward pass to get KV cache
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True
            )
            
            # Compress and chunk KV cache
            kv_cache = outputs.past_key_values
            chunks = self.chunk_manager.split_kv_cache(kv_cache, content=str(input_ids.tolist()))
            
            # Track for reuse
            gen_id = self.chunk_manager.generate_chunk_id(
                str(input_ids.tolist()), 0, input_ids.shape[1]
            )
            self._active_generations[gen_id] = {
                'chunks': chunks,
                'last_token': input_ids[:, -1:]
            }
            
            # Generate tokens
            cur_len = input_ids.shape[1]
            max_length = gen_kwargs.get('max_length', cur_len + 100)
            
            while cur_len < max_length:
                # Decode cached KV chunks asynchronously
                async for chunk_id, decoded_tensors in self.pipeline.decode_chunks_pipelined(
                    [chunk['id'] for chunk in chunks],
                    storage_path='cache'
                ):
                    # Use decoded KV cache for next token prediction
                    outputs = self.model(
                        input_ids=self._active_generations[gen_id]['last_token'],
                        attention_mask=attention_mask[:, -1:],
                        past_key_values=decoded_tensors,
                        use_cache=True,
                        return_dict=True
                    )
                    
                    # Sample next token
                    next_token = self.sample_next_token(
                        outputs.logits[:, -1, :],
                        temperature=gen_kwargs.get('temperature', 0.7),
                        top_p=gen_kwargs.get('top_p', 0.9)
                    )
                    
                    # Update state
                    self._active_generations[gen_id]['last_token'] = next_token
                    attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=1)
                    cur_len += 1
                    
                    # Compress and chunk new KV cache
                    chunks = self.chunk_manager.split_kv_cache(
                        outputs.past_key_values,
                        content=str(next_token.tolist())
                    )
                    self._active_generations[gen_id]['chunks'] = chunks
                    
                    # Yield token
                    yield self.tokenizer.decode(next_token[0])
                    
                    if next_token[0] == self.tokenizer.eos_token_id:
                        break
                        
        except KVCacheError as e:
            # Fallback to standard generation if cache fails
            print(f"KV cache error, falling back to standard generation: {str(e)}")
            for token in self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            ):
                yield self.tokenizer.decode(token)
                
        finally:
            # Cleanup
            if gen_id in self._active_generations:
                del self._active_generations[gen_id]
            
            # Record metrics
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to ms
            metrics.record_latency(latency, use_cachegen)
            metrics.record_memory(use_cachegen)
    
    def sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """Sample next token using temperature and nucleus sampling."""
        if temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
            
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        
        # Nucleus sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        nucleus = cumsum_probs < top_p
        nucleus = torch.cat([nucleus.new_ones(1), nucleus[:-1]], dim=0)
        
        # Sample from filtered distribution
        filtered_probs = torch.where(nucleus, sorted_probs, 0.0)
        filtered_probs = filtered_probs / filtered_probs.sum()
        next_token = torch.multinomial(filtered_probs, num_samples=1)
        return sorted_indices[0, next_token]

# Initialize FastAPI app
app = FastAPI(title="LLM Service with CacheGen")
app.include_router(metrics.router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
llm_service = None

@app.on_event("startup")
async def startup_event():
    global llm_service
    llm_service = LLMService()

@app.post("/generate")
async def generate(prompt: str, use_cachegen: bool = True):
    start_time = time.time()
    # Tokenize input
    inputs = llm_service.tokenizer(
        prompt,
        return_tensors="pt",
        padding=True
    ).to(llm_service.device)
    
    # Stream response
    async def generate_stream():
        async for token in llm_service._stream_generate(
            inputs.input_ids,
            inputs.attention_mask,
            {
                'max_length': request.max_length,
                'temperature': request.temperature,
                'top_p': request.top_p
            }
        ):
            yield f"data: {token}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream"
    )

@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """WebSocket endpoint for text generation."""
    await websocket.accept()
    
    try:
        while True:
            # Receive request
            data = await websocket.receive_json()
            request = GenerationRequest(**data)
            
            # Tokenize input
            inputs = llm_service.tokenizer(
                request.prompt,
                return_tensors="pt",
                padding=True
            ).to(llm_service.device)
            
            # Stream tokens
            async for token in llm_service._stream_generate(
                inputs.input_ids,
                inputs.attention_mask,
                {
                    'max_length': request.max_length,
                    'temperature': request.temperature,
                    'top_p': request.top_p
                }
            ):
                await websocket.send_text(token)
                
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()
