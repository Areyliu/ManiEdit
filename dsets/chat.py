"""Chat templates (without the answer) used to build editing and evaluation prompts."""


def model_family(model_name: str) -> str:
    name = model_name.lower()
    if "qwen" in name:
        return "qwen"
    if "llama" in name:
        return "llama"
    raise ValueError(f"Only Qwen and Llama-3 chat models are supported, got {model_name}")


def wrap_question(que: str, family: str) -> str:
    if family == "qwen":
        return f"<|im_start|>user\n{que}<|im_end|>\n<|im_start|>assistant\n"
    return (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def end_of_turn(family: str) -> str:
    return "<|im_end|>" if family == "qwen" else "<|eot_id|>"


def strip_end_of_turn(answer: str) -> str:
    for eot in ("<|im_end|>", "<|eot_id|>"):
        if answer.endswith(eot):
            return answer[: -len(eot)]
    return answer
