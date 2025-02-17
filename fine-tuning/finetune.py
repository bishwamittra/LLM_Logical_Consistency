import os
# change the cache appropriately
# cache_path = "/NS/llm-1/nobackup/bishwa/huggingface_hub_cache"
# os.environ['HF_HOME'] = cache_path 
os.environ["WANDB_MODE"] = "offline"
import torch
import wandb
import gc
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    # HfArgumentParser,
    TrainingArguments,
    # pipeline,
    # logging,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
from utils_finetuning import get_args, get_dataset, print_trainable_parameters, count_trainable_params, find_all_linear_names
from datetime import date








if __name__=="__main__":
    # Define a custom callback to print validation loss during training
    # Set up the callback
    # validation_callback = ValidationCallback()
    # Load dataset (you can process it here)
    #dataset = load_dataset(dataset_name, split="train")


    # print(torch.cuda.device_count())
    # print(torch.cuda.device_ids())
    # for i in range(torch.cuda.device_count()):
    #     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    args = get_args()

    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16 = False
    bf16 = False
    current_device = Accelerator().process_index
    print(current_device)
    device_map = {"": current_device}

    # device_map = "auto" # device_map = {"": 0}
    # device_map={'':torch.cuda.current_device()}

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    if(args.use_4bit):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            bf16 = True
            print("Using bfloat16 for training")
            print("=" * 80)

    # # Load base model
    # model = AutoModelForCausalLM.from_pretrained(
    #     # args.model_name,
    #     f"{args.model_path}",
    #     device_map=device_map
    # )
    # model.config.use_cache = False
    # model.config.pretraining_tp = 1


    # print_trainable_parameters(model)
    # count_trainable_params(model)
    

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        # args.model_name,
        f"{args.model_path}",
        quantization_config=bnb_config if args.use_4bit else None,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1


    print_trainable_parameters(model)
    count_trainable_params(model)

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"{args.model_path}")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    if(args.use_4bit):
        # Load LoRA configuration
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=find_all_linear_names(model),
        )

    # # delete
    # model = get_peft_model(model, peft_config) if args.use_4bit else model
    # model.print_trainable_parameters()
    # ================================================================================
    # Your GPU supports bfloat16: accelerate training with bf16=True
    # Using bfloat16 for training
    # ================================================================================
    # Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.30it/s]
    # trainable params: 2506172416 || all params: 2506172416 || trainable%: 100.0
    # Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.94it/s]
    # trainable params: 524363776 || all params: 1515268096 || trainable%: 34.605346564361376
    # trainable params: 78,446,592 || all params: 1,593,714,688 || trainable%: 4.922248165915128
    # quit()


    # setup wandb
    today = date.today()
    wandb_project_name = f"Consistency Finetuning"
    wandb_service_wait=300
    os.environ["WANDB_PROJECT"] = wandb_project_name
    os.environ["WANDB_WATCH"] = "all"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # use your own api
    os.environ["WANDB_API_KEY"]="cd8d79cbe96f9d1e1d5fbf1b4829ee38e3e2f76f"
    run_name = f"finetuning | {args.model_path.split('/')[-1]}  | {today.strftime('%d-%b-%Y')}"
    wandb.init(project=wandb_project_name)
    wandb.run.name = run_name
    params_dict = vars(args)
    wandb.config.update(params_dict)




    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        group_by_length=args.group_by_length,
        lr_scheduler_type=args.lr_scheduler_type,
        evaluation_strategy="epoch",
        report_to=["wandb"],
        save_strategy="epoch"


        # save_steps=args.save_steps,
        # report_to="tensorboard",
        # num_training_steps=600,
        # evaluation_strategy="steps",  # or "epoch" for evaluation at the end of each epoch
        # eval_steps=val_steps,  # specify evaluation steps
        # eval_dataset=val_dataset  # specify the validation dataset
        # Add evaluation strategy with validation dataset
        # evaluation_strategy="steps",  # or "epoch" for evaluation at the end of each epoch
        # eval_steps=25,  # specify evaluation steps
        # eval_dataset=val_dataset  # specify the validation dataset

    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=get_dataset(filenames=args.train_filename, query_types=args.query_type, model_path=args.model_path, nrows=args.train_nrows),
        eval_dataset=get_dataset(filenames=args.eval_filename, query_types=args.query_type, model_path=args.model_path, nrows=args.eval_nrows),
        peft_config=peft_config if args.use_4bit else None,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=args.packing,
        # callbacks=[validation_callback],  # Add the validation callback
    )

    # Train model
    trainer.train()
    # Store model weights after training
    # os.system(f"rm -r {args.model_path_fine_tuned}")
    # trainer.save_model(f"{args.model_path_fine_tuned}/finetuned_weights")

    # release memory
    model = None
    tokenizer = None
    trainer = None
    gc.collect()
    torch.cuda.empty_cache()


    # mv everything to NFS
    # os.system(f"mkdir -p {args.output_dir.replace('/tmp/', '')}")
    # os.system(f"mv {args.output_dir}/* {args.output_dir.replace('/tmp/', '')}")



    """
        Saving
    """
    if(False):
        # Reload model in FP16 and merge it with LoRA weights
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            # low_cpu_mem_usage=True,
            # return_dict=True,
            # torch_dtype=torch.float16,
            device_map=device_map,
        )
        model = PeftModel.from_pretrained(base_model, f"{args.model_path_fine_tuned}/finetuned_weights")
        model = model.merge_and_unload()

        # Reload tokenizer to save it
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # finally save the model and tokenizer
        model.save_pretrained(args.model_path_fine_tuned)
        tokenizer.save_pretrained(args.model_path_fine_tuned)

    wandb.finish()
