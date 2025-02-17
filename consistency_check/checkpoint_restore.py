
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("finetuned_model_path", type=str, help="Path to the finetuned model")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint")
    parser.add_argument("--base_model_path", type=str, default=None, help="Path to the pretrained model")
    args = parser.parse_args()

    assert args.checkpoint_path is not None
    assert args.base_model_path is not None

    # Reload model in FP16 and merge it with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        # low_cpu_mem_usage=True,
        # return_dict=True,
        # torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.checkpoint_path)
    model = model.merge_and_unload()

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # finally save the model and tokenizer, which we will use via vllm
    model.save_pretrained(args.finetuned_model_path)
    tokenizer.save_pretrained(args.finetuned_model_path)
    