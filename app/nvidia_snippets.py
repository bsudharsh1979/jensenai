"""Verified NVIDIA library snippets the model may adapt for the "Architecture & code" section.

The generator is told to pick the closest snippet and change only model names / prompts / sizes, so code in
answers uses real, documented public APIs instead of invented ones. Keep each entry short.
"""

SNIPPETS: dict[str, dict] = {
    "tensorrt_llm": {
        "when": "GPU inference, Blackwell/Rubin/NVL72 serving, NVFP4, MLPerf, high-throughput LLM serving",
        "lang": "python",
        "code": '''# TensorRT-LLM LLM API (pip install tensorrt-llm) — https://nvidia.github.io/TensorRT-LLM/
from tensorrt_llm import LLM, SamplingParams

llm = LLM(model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", tensor_parallel_size=1)
params = SamplingParams(temperature=0.2, max_tokens=256)
for out in llm.generate(["Summarise disaggregated inference in two sentences."], params):
    print(out.outputs[0].text)''',
    },
    "dynamo": {
        "when": "NVIDIA Dynamo, disaggregated prefill/decode, KV-cache routing, inference at data-center scale",
        "lang": "bash",
        "code": '''# NVIDIA Dynamo (pip install "ai-dynamo[all]") — https://github.com/ai-dynamo/dynamo
# Front-end + router, then separate prefill and decode workers (disaggregated serving)
python -m dynamo.frontend --http-port 8000 &
python -m dynamo.vllm --model Qwen/Qwen3-8B --is-prefill-worker &
python -m dynamo.vllm --model Qwen/Qwen3-8B &
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \\
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"hello"}],"max_tokens":64}' ''',
    },
    "nim": {
        "when": "NVIDIA NIM microservices, enterprise deployment, OpenAI-compatible endpoints, AI Enterprise",
        "lang": "bash",
        "code": '''# NVIDIA NIM: pull and run an LLM microservice, then call it with the OpenAI-compatible API — https://docs.nvidia.com/nim/
export NGC_API_KEY=<your key>
docker run --gpus all -p 8000:8000 -e NGC_API_KEY \\
  nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2:latest
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \\
  -d '{"model":"nvidia/nvidia-nemotron-nano-9b-v2","messages":[{"role":"user","content":"What is NIM?"}]}' ''',
    },
    "nemo_automodel": {
        "when": "fine-tuning, LoRA/SFT, Nemotron training, NeMo framework, DGX Spark training playbooks",
        "lang": "bash",
        "code": '''# NeMo AutoModel LoRA fine-tune (official DGX Spark playbook) — https://github.com/NVIDIA-NeMo/Automodel
docker run --gpus all -it --rm -v $PWD:/work nvcr.io/nvidia/nemo-automodel:26.08 \\
  automodel finetune llm -c /work/lora.yaml
# lora.yaml (excerpt): model.pretrained_model_name_or_path: nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
#   peft: {peft_fn: nemo_automodel.components._peft.lora.PeftConfig, dim: 16, alpha: 32, dropout: 0.05}
#   dataset: {_target_: nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset, path: /work/train.jsonl}''',
    },
    "cuda_python": {
        "when": "CUDA kernels, GPU compute fundamentals, GB10/Grace Blackwell programming, accelerated computing",
        "lang": "python",
        "code": '''# CUDA Python via CuPy / Numba (pip install cupy-cuda13x numba) — https://developer.nvidia.com/cuda-python
import cupy as cp
from numba import cuda

@cuda.jit
def saxpy(a, x, y, out):
    i = cuda.grid(1)
    if i < x.size:
        out[i] = a * x[i] + y[i]

x = cp.random.rand(1 << 24, dtype=cp.float32); y = cp.random.rand(1 << 24, dtype=cp.float32)
out = cp.empty_like(x)
saxpy[(x.size + 255) // 256, 256](2.0, x, y, out)
print(float(out[:3].sum()))''',
    },
    "rapids": {
        "when": "data analytics, cuDF/RAPIDS, GPU dataframes, ETL acceleration, pandas acceleration",
        "lang": "python",
        "code": '''# RAPIDS cuDF — GPU DataFrames with the pandas API (pip install cudf-cu13) — https://rapids.ai
import cudf

df = cudf.read_parquet("transactions.parquet")
by_region = df.groupby("region").agg({"amount": ["sum", "mean"], "customer_id": "nunique"})
print(by_region.sort_values(("amount", "sum"), ascending=False).head())''',
    },
    "isaac_sim": {
        "when": "robotics, humanoids, physical AI, Isaac Sim / Isaac Lab, simulation, GR00T",
        "lang": "python",
        "code": '''# Isaac Sim 5 standalone script — https://docs.isaacsim.omniverse.nvidia.com
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(get_assets_root_path() + "/Isaac/Robots/Franka/franka.usd", "/World/Franka")
world.reset()
for _ in range(240):
    world.step(render=False)
app.close()''',
    },
    "cosmos": {
        "when": "world foundation models, Cosmos, synthetic video data for robotics / autonomous vehicles",
        "lang": "bash",
        "code": '''# NVIDIA Cosmos world foundation models — https://github.com/nvidia-cosmos
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5 && cd cosmos-predict2.5
hf download nvidia/Cosmos-Predict2.5-2B --local-dir checkpoints/Cosmos-Predict2.5-2B
python examples/inference.py --model 2B --prompt "A warehouse robot lifts a pallet onto a shelf" \\
  --input_path assets/warehouse_frame.jpg --save_path outputs/warehouse.mp4''',
    },
    "nemotron_hf": {
        "when": "Nemotron open models, running an open model locally, Hugging Face Transformers, DGX Spark local inference",
        "lang": "python",
        "code": '''# Run an open Nemotron model with Transformers (pip install transformers torch) — https://huggingface.co/nvidia
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
msgs = [{"role": "user", "content": "What is an AI factory?"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
print(tok.decode(model.generate(ids, max_new_tokens=200)[0][ids.shape[1]:], skip_special_tokens=True))''',
    },
    "omniverse": {
        "when": "digital twins, Omniverse, OpenUSD, industrial simulation, factory digital twins",
        "lang": "python",
        "code": '''# OpenUSD scene authoring for Omniverse digital twins (pip install usd-core) — https://openusd.org
from pxr import Usd, UsdGeom, Gf

stage = Usd.Stage.CreateNew("factory_twin.usda")
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
cell = UsdGeom.Xform.Define(stage, "/Factory/Cell_01")
robot = UsdGeom.Cube.Define(stage, "/Factory/Cell_01/RobotBase")
robot.AddTranslateOp().Set(Gf.Vec3d(2.0, 0.0, 0.5))
stage.GetRootLayer().Save()''',
    },
}


def catalog() -> str:
    parts = ["Approved NVIDIA code snippets (documented public APIs). Pick the ONE most relevant to the question, adapt only "
             "model names, prompts, paths or sizes, and keep the source URL comment. Do not invent CLIs, flags or functions "
             "that are not in these snippets:"]
    for key, s in SNIPPETS.items():
        parts.append(f"--- {key} (use for: {s['when']}) ---\n```{s['lang']}\n{s['code']}\n```")
    return "\n".join(parts)


def pick(text: str) -> tuple[str, dict]:
    """Choose the catalog snippet whose 'when' keywords best match the question/answer text."""
    import re as _re
    words = set(_re.findall(r"[a-z0-9][a-z0-9\-\.]+", text.lower()))
    best, best_score = "nemotron_hf", -1
    for key, sn in SNIPPETS.items():
        kws = {w.strip(" ,/()").lower() for w in _re.split(r"[,/]| and ", sn["when"]) if w.strip()}
        score = sum(1 for k in kws if k and (k in text.lower()))
        score += sum(1 for k in key.split("_") if k in words)
        if score > best_score:
            best, best_score = key, score
    return best, SNIPPETS[best]
