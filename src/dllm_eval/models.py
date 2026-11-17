"""Small, explicit generation adapters for the four assigned models."""
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "qwen": ("Qwen/Qwen2.5-7B-Instruct", "ar", "a09a35458c702b33eeacc393d103063234e8bc28"),
    "llama": ("meta-llama/Meta-Llama-3-8B-Instruct", "ar", "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"),
    "llama31": ("meta-llama/Llama-3.1-8B-Instruct", "ar", "0e9e39f249a16976918f6564b8830bc894c89659"),
    "llada": ("GSAI-ML/LLaDA-8B-Instruct", "llada", "08b83a6feb34df1a6011b80c3c00c7563e963b07"),
    "dream": ("Dream-org/Dream-v0-Instruct-7B", "dream", "05334cb9faaf763692dcf9d8737c642be2b2a6ae"),
}


class Generator:
    def __init__(self, name):
        if name not in MODELS:
            raise ValueError(f"unknown model {name!r}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the 7–8B model baseline")
        self.name = name
        self.repo, self.kind, self.revision = MODELS[name]
        remote = self.kind != "ar"
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo, revision=self.revision, trust_remote_code=remote)
        cls = AutoModelForCausalLM if self.kind == "ar" else AutoModel
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = cls.from_pretrained(self.repo, revision=self.revision, trust_remote_code=remote,
                                         torch_dtype=dtype, low_cpu_mem_usage=True).to("cuda").eval()

    def generate(self, prompt, new_tokens=32, steps=16, chat=True):
        tokenizer = self.tokenizer
        if chat:
            inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                   add_generation_prompt=True, return_tensors="pt")
        else:
            inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
            if tokenizer.bos_token_id is not None and inputs[0, 0].item() != tokenizer.bos_token_id:
                inputs = torch.cat([torch.tensor([[tokenizer.bos_token_id]], dtype=inputs.dtype), inputs], dim=1)
        inputs = inputs.to("cuda")
        with torch.inference_mode():
            if self.kind == "ar":
                output = self.model.generate(inputs, max_new_tokens=new_tokens, do_sample=False,
                                             pad_token_id=tokenizer.eos_token_id)
            elif self.kind == "dream":
                output = self.model.diffusion_generate(inputs, attention_mask=torch.ones_like(inputs),
                                                        max_new_tokens=new_tokens, steps=min(steps, new_tokens),
                                                        temperature=0.0, alg="entropy",
                                                        return_dict_in_generate=True).sequences
            else:
                output = self._llada_generate(inputs, new_tokens, min(steps, new_tokens))
        decoded = tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True)
        if tokenizer.eos_token and tokenizer.eos_token in decoded:
            decoded = decoded.split(tokenizer.eos_token, 1)[0]
        return decoded.strip()

    def _llada_generate(self, prompt, length, steps):
        """Low-confidence remasking, based on the official LLaDA generate.py algorithm."""
        mask_id = getattr(self.model.config, "mask_token_id", 126336)
        if mask_id is None:
            mask_id = 126336
        x = torch.cat([prompt, torch.full((1, length), mask_id, device=prompt.device, dtype=prompt.dtype)], 1)
        base, rem = divmod(length, steps)
        for step in range(steps):
            mask = x == mask_id
            logits = self.model(x).logits[:, -length:, :]
            probabilities = logits.softmax(dim=-1)
            confidence, predicted = probabilities.max(dim=-1)
            confidence = confidence.masked_fill(~mask[:, -length:], -float("inf"))
            count = base + (step < rem)
            chosen = confidence.topk(count, dim=1).indices + prompt.shape[1]
            x.scatter_(1, chosen, predicted.gather(1, chosen - prompt.shape[1]))
        if (x[:, -length:] == mask_id).any():
            raise RuntimeError("LLaDA left mask tokens unfilled")
        return x
