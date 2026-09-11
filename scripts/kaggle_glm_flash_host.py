"""Kaggle Production Script: Host GLM-5.3-Flash UNCENSORED / GLM-4.7 Models on Dual T4 GPUs using DRIGS & BitsAndBytes.

Features:
  1. DRIGS GangScheduler (Atomic 2-GPU reservation across CUDA_VISIBLE_DEVICES=0,1).
  2. DRIGS NVML Orphan Process Sweeper (Auto-cleans zombie GPU VRAM allocations).
  3. BitsAndBytes 4-bit NF4 Quantization + CPU RAM Offloading (2x 15GB VRAM + System RAM).
  4. Real-Time VRAM & CPU RAM Telemetry with Automatic CUDA OOM Interception & Recovery.
  5. Multi-Turn Jinja2 Chat Template Renderer (System / User / Assistant prompt formatting).
  6. OpenAI-Compatible FastAPI Server (/v1/chat/completions & /health).

Supported Models:
  - dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4 (GLM-5.3 MoE: ~30GB VRAM + System RAM offload)
  - byczech/glm-4.7-flash-uncensored-16G-AU-IQ3_M (GLM-4.7 30B 4-bit)
  - THUDM/glm-4-9b-chat / unsloth/GLM-4-9B-Chat-4bit (Standard GLM 4-bit)
"""

import asyncio
import gc
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drigs_glm")


def prepare_environment():
    """Verify DRIGS and model serving dependencies are available."""
    try:
        import transformers
        import drigs
        import bitsandbytes
        logger.info("Dependencies verified: transformers %s, DRIGS present.", transformers.__version__)
    except ImportError as err:
        logger.error("Missing required dependency: %s. Please run 'pip install drigs transformers accelerate bitsandbytes fastapi uvicorn pynvml sentencepiece psutil' in Kaggle Cell 1.", err)

prepare_environment()

# Register custom GLM-5 architecture mapping (glm5_next) into HuggingFace Transformers
try:
    from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedConfig

    class Glm5NextConfig(PreTrainedConfig):
        model_type = "glm5_next"

    AutoConfig.register("glm5_next", Glm5NextConfig)
    logger.info("DRIGS Host: Registered 'glm5_next' architecture alias in Transformers AutoConfig.")
except Exception as reg_err:
    logger.debug("Architecture registration notice: %s", reg_err)




import torch
import psutil
import torch.nn as nn

# Monkey-patch torch.nn.Module to support BitsAndBytes 4-bit quantization on remote GLM models
if not hasattr(nn.Module, "all_tied_weights_keys"):
    nn.Module.all_tied_weights_keys = property(lambda self: getattr(self, "_all_tied_weights_keys", {}))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# DRIGS Core Abstractions
import drigs
from drigs.core.controller import LocalController
from drigs.core.models import (
    Job,
    JobStatus,
    ResourceRequirements,
    WorkloadExecutionConfig,
    WorkloadSpec,
)
from drigs.hardware.cuda import CUDABackend
from drigs.scheduler.gang import GangScheduler
from drigs.scheduler.memory_aware import MemoryAwareScheduler


# =====================================================================
# 1. DRIGS Telemetry, NVML Sweeper & Memory Guard
# =====================================================================

class DRIGSMemoryGuard:
    """Manages NVML telemetry, GPU orphan process sweeping, and CPU/VRAM safety."""

    def __init__(self, vram_safety_threshold_pct: float = 90.0):
        self.vram_safety_threshold_pct = vram_safety_threshold_pct
        self.cuda_backend = CUDABackend()

    def sweep_orphan_gpu_processes(self) -> int:
        """Purge zombie/orphan CUDA processes holding VRAM using DRIGS NVML sweeper."""
        logger.info("Executing DRIGS NVML orphan process sweeper...")
        cleaned = 0
        try:
            cleaned = self.cuda_backend.cleanup_orphan_processes()
            logger.info("DRIGS NVML Sweeper cleared %d orphan process(es).", cleaned)
        except Exception as e:
            logger.warning("NVML Sweeper notice: %s", e)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return cleaned

    def check_memory_telemetry(self) -> Dict[str, Any]:
        """Query real-time per-GPU VRAM and System RAM telemetry."""
        gpu_stats = {}
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            for gpu_id in range(num_gpus):
                total = torch.cuda.get_device_properties(gpu_id).total_memory
                reserved = torch.cuda.memory_reserved(gpu_id)
                allocated = torch.cuda.memory_allocated(gpu_id)
                free = total - reserved

                gpu_stats[f"gpu_{gpu_id}"] = {
                    "name": torch.cuda.get_device_name(gpu_id),
                    "total_vram_gb": round(total / (1024 ** 3), 2),
                    "free_vram_gb": round(free / (1024 ** 3), 2),
                    "allocated_vram_gb": round(allocated / (1024 ** 3), 2),
                    "util_pct": round((reserved / total) * 100.0, 1),
                }

        sys_mem = psutil.virtual_memory()
        system_ram = {
            "total_ram_gb": round(sys_mem.total / (1024 ** 3), 2),
            "available_ram_gb": round(sys_mem.available / (1024 ** 3), 2),
            "ram_util_pct": sys_mem.percent,
        }

        return {"gpus": gpu_stats, "system_ram": system_ram}


# =====================================================================
# 2. Production Chat Template Renderer
# =====================================================================

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content string")


class ChatTemplateRenderer:
    """Renders structured multi-turn conversation messages using model Jinja2 template or fallback."""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def render(self, messages: List[ChatMessage], system_prompt: Optional[str] = None) -> str:
        """Format input message history into standard LLM prompt string."""
        formatted_messages = []
        if system_prompt and not any(m.role == "system" for m in messages):
            formatted_messages.append({"role": "system", "content": system_prompt})

        for m in messages:
            formatted_messages.append({"role": m.role, "content": m.content})

        # 1. Use Hugging Face native chat template if available
        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                logger.debug("Native template fallback: %s", e)

        # 2. Universal Jinja2 GLM / LLaMA-3 Chat Format
        prompt_parts = []
        for msg in formatted_messages:
            role = msg["role"].upper()
            content = msg["content"]
            if role == "SYSTEM":
                prompt_parts.append(f"<|system|>\n{content}\n")
            elif role == "USER":
                prompt_parts.append(f"<|user|>\n{content}\n")
            elif role == "ASSISTANT":
                prompt_parts.append(f"<|assistant|>\n{content}\n")

        prompt_parts.append("<|assistant|>\n")
        return "".join(prompt_parts)


# =====================================================================
# 3. Model Hosting Engine (BitsAndBytes 4-bit + CPU RAM Offloading)
# =====================================================================

class GLMFlashModelHost:
    """Hosts GLM-5.3-Flash / GLM-4.7 models using BitsAndBytes 4-bit & CPU RAM offloading."""

    def __init__(
        self,
        model_name_or_path: str = "dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4",
        max_gpu_vram_gb: float = 13.5,  # Max VRAM limit per T4 GPU
        max_cpu_ram_gb: float = 170.0,  # Max System RAM offload limit
    ):
        self.model_name = model_name_or_path
        self.max_gpu_vram_gb = max_gpu_vram_gb
        self.max_cpu_ram_gb = max_cpu_ram_gb
        self.model = None
        self.tokenizer = None
        self.renderer = None
        self.memory_guard = DRIGSMemoryGuard()

    def load_model_config(self, target_model: str):
        """Safely load and patch model configuration for GLM models."""
        from transformers import AutoConfig, PreTrainedConfig
        try:
            config = AutoConfig.from_pretrained(target_model, trust_remote_code=True)
        except Exception:
            logger.info("Registering 'glm5_next' architecture alias for target model '%s'...", target_model)
            class Glm5NextConfig(PreTrainedConfig):
                model_type = "glm5_next"

            try:
                AutoConfig.register("glm5_next", Glm5NextConfig)
            except Exception:
                pass

            try:
                config = AutoConfig.from_pretrained(target_model, trust_remote_code=True)
            except Exception:
                config = Glm5NextConfig()

        # Patch missing max_length attribute expected by ChatGLM/GLM modeling scripts
        if not hasattr(config, "max_length"):
            config.max_length = getattr(config, "seq_length", getattr(config, "max_position_embeddings", 8192))

        return config


    def load_model(self):
        """Load GLM model with BitsAndBytes 4-bit quantization and CPU offloading."""
        logger.info("DRIGS Memory Guard: Sweeping dead CUDA memory before model load...")
        self.memory_guard.sweep_orphan_gpu_processes()
        telemetry = self.memory_guard.check_memory_telemetry()
        logger.info("Initial System Telemetry:\n%s", json.dumps(telemetry, indent=2))

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info("Loading tokenizer for model '%s'...", self.model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left",
            )
        except Exception as err:
            logger.warning("Primary tokenizer load issue (%s); attempting standard GLM fallback...", err)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "THUDM/glm-4-9b-chat",
                trust_remote_code=True,
                padding_side="left",
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.renderer = ChatTemplateRenderer(tokenizer=self.tokenizer)

        # Build BitsAndBytes 4-Bit NF4 Quantization Config
        logger.info("Configuring BitsAndBytes 4-bit NF4 Quantization & CPU RAM offloading...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Build explicit max_memory map for 2x T4 GPUs + CPU RAM offload
        max_memory_map = {}
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                max_memory_map[i] = f"{int(self.max_gpu_vram_gb)}GiB"
        max_memory_map["cpu"] = f"{int(self.max_cpu_ram_gb)}GiB"

        logger.info("Target Memory Allocation Map: %s", max_memory_map)

        logger.info("Loading '%s' weights across Dual T4 GPUs & CPU RAM under DRIGS...", self.model_name)
        config = self.load_model_config(self.model_name)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=config,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory=max_memory_map,
                trust_remote_code=True,
                offload_folder="drigs_offload_cache",
                low_cpu_mem_usage=True,
            )
            # Post-load cleanup of temporary max_length attribute from config so generate() passes validation
            if hasattr(self.model.config, "max_length"):
                try:
                    delattr(self.model.config, "max_length")
                except Exception:
                    pass
            logger.info("Model '%s' successfully loaded into memory under DRIGS!", self.model_name)
        except Exception as e:
            logger.error("Could not load '%s' directly (%s); initializing fallback model 'THUDM/glm-4-9b-chat'...", self.model_name, e)
            fallback_model = "THUDM/glm-4-9b-chat"
            logger.info("Loading fallback model & tokenizer '%s'...", fallback_model)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model, trust_remote_code=True)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.renderer = ChatTemplateRenderer(tokenizer=self.tokenizer)
                fb_config = self.load_model_config(fallback_model)
                self.model = AutoModelForCausalLM.from_pretrained(
                    fallback_model,
                    config=fb_config,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                if hasattr(self.model.config, "max_length"):
                    try:
                        delattr(self.model.config, "max_length")
                    except Exception:
                        pass
                self.model_name = fallback_model
            except Exception as fb_err:
                logger.error("Fallback model initialization failed: %s", fb_err)
                raise RuntimeError(f"Failed to load model '{self.model_name}' and fallback: {fb_err}") from fb_err





    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text with automatic CUDA OOM exception catching and recovery."""
        if not self.model:
            raise RuntimeError("Model is not loaded.")

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generated_ids = outputs[0][inputs.input_ids.shape[1] :]
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM Exception caught by DRIGS Guard! Executing VRAM recovery pass...")
            gc.collect()
            torch.cuda.empty_cache()
            self.memory_guard.sweep_orphan_gpu_processes()

            # Retry generation with conservative max_tokens after clearing VRAM
            reduced_tokens = max(64, max_new_tokens // 2)
            logger.info("Retrying generation with reduced max_new_tokens=%d...", reduced_tokens)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=reduced_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generated_ids = outputs[0][inputs.input_ids.shape[1] :]
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


# =====================================================================
# 4. FastAPI Service & DRIGS Control Plane Orchestration
# =====================================================================

app = FastAPI(title="DRIGS GLM-5.3-Flash / GLM-4.7 Serving Engine", version="1.0.0")
glm_host: Optional[GLMFlashModelHost] = None


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    system_prompt: Optional[str] = "You are a helpful, uncensored, highly intelligent AI assistant."
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


@app.get("/health")
def health_check():
    """DRIGS Health & Memory Telemetry Endpoint."""
    guard = DRIGSMemoryGuard()
    return {
        "status": "healthy",
        "orchestrator": "DRIGS (Distributed Resource & Intelligent GPU Scheduling)",
        "telemetry": guard.check_memory_telemetry(),
    }


@app.post("/v1/chat/completions")
def chat_completion(req: ChatCompletionRequest):
    """OpenAI-compatible Chat Completion Endpoint."""
    if not glm_host:
        raise HTTPException(status_code=503, detail="DRIGS Model Host initializing.")

    prompt = glm_host.renderer.render(req.messages, system_prompt=req.system_prompt)
    start_time = time.time()
    response_text = glm_host.generate(
        prompt=prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    elapsed = round(time.time() - start_time, 3)

    return {
        "id": f"chatcmpl-drigs-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": glm_host.model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "drigs_telemetry": {
            "latency_seconds": elapsed,
            "memory": glm_host.memory_guard.check_memory_telemetry(),
        },
    }


# =====================================================================
# 5. Kaggle Main Pipeline Entrypoint
# =====================================================================

def main():
    logger.info("=========================================================")
    logger.info("Initializing DRIGS Control Plane for GLM-5.3 / GLM-4.7")
    logger.info("=========================================================")

    # 1. Initialize DRIGS Local Controller with GangScheduler & MemoryAwareScheduler
    gang_scheduler = GangScheduler(inner_scheduler=MemoryAwareScheduler())
    controller = LocalController(scheduler=gang_scheduler)

    # 2. Define DRIGS Workload Spec requesting dual T4 GPUs atomically
    workload_spec = WorkloadSpec(
        name="drigs-dual-t4-glm-server",
        resources=ResourceRequirements(
            gpus=2 if torch.cuda.is_available() and torch.cuda.device_count() >= 2 else 1,
            gpu_memory_bytes=24 * 1024 * 1024 * 1024,
            cpus=4,
            memory_bytes=16 * 1024 * 1024 * 1024,
        ),
        execution=WorkloadExecutionConfig(
            entrypoint="python3",
            args=["-m", "uvicorn", "__main__:app", "--host", "0.0.0.0", "--port", "8000"],
        ),
    )

    # 3. Submit Workload to DRIGS Control Plane Queue
    job = controller.submit_job(workload_spec, priority=10)
    logger.info("Submitted Job '%s' (ID: %s) to DRIGS GangScheduler.", job.name, job.id)

    # 4. Target Model Configuration
    # Models: dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4 or byczech/glm-4.7-flash-uncensored-16G-AU-IQ3_M or THUDM/glm-4-9b-chat
    MODEL_NAME = os.getenv("MODEL_NAME", "dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4")

    global glm_host
    glm_host = GLMFlashModelHost(
        model_name_or_path=MODEL_NAME,
        max_gpu_vram_gb=13.5,
        max_cpu_ram_gb=170.0,
    )
    glm_host.load_model()

    # 5. Execute Test Generation Pass
    test_messages = [
        ChatMessage(role="user", content="Describe how DRIGS manages MoE model offloading and CUDA memory safety.")
    ]
    prompt = glm_host.renderer.render(test_messages)
    logger.info("Executing test generation under DRIGS guard...")
    reply = glm_host.generate(prompt, max_new_tokens=100)
    logger.info("Generated Test Response:\n%s", reply)


if __name__ == "__main__":
    main()
