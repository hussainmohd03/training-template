import torch
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class ModelTrainer:

    def __init__(self, config: dict):
        self.model_config = config["model"]
        self.lora_cfg = config["lora"]
        self.train_config = config["training"]
        self.data_config = config["data"]

        self.model = None
        self.tokenizer = None
        self.trainer = None

    def _build_bnb_config(self):
        # Only build a quantization config if quantization is requested
        if not self.model_config.get("load_in_4bit"):
            return None

        compute_dtype = (
            torch.bfloat16
            if self.model_config.get("bnb_4bit_compute_dtype") == "bfloat16"
            else torch.float16
        )
        return BitsAndBytesConfig(
            load_in_4bit=self.model_config["load_in_4bit"],
            bnb_4bit_quant_type=self.model_config["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=self.model_config["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=compute_dtype,
        )

    def load_model_and_tokenizer(self):
        bnb_config = self._build_bnb_config()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config["base_model"],
            quantization_config=bnb_config,   # None when not quantizing
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["base_model"],
            trust_remote_code=True,
        )
        return self.model, self.tokenizer

    def apply_lora(self):
        peft_config = LoraConfig(
            r=self.lora_cfg["r"],
            lora_alpha=self.lora_cfg["lora_alpha"],
            target_modules=self.lora_cfg["target_modules"],
            lora_dropout=self.lora_cfg["lora_dropout"],
            bias=self.lora_cfg["bias"],
            task_type=(
                TaskType.CAUSAL_LM
                if self.lora_cfg["task_type"] == "CAUSAL_LM"
                else TaskType.SEQ_2_SEQ_LM
            ),
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()  
        return self.model

    def _build_sft_config(self):
        tc = self.train_config
        return SFTConfig(
            output_dir=self.model_config["output_dir"],
            completion_only_loss=tc["completion_only_loss"],
            num_train_epochs=tc["num_train_epochs"],
            learning_rate=tc["learning_rate"],
            per_device_train_batch_size=tc["per_device_train_batch_size"],
            gradient_accumulation_steps=tc["gradient_accumulation_steps"],
            warmup_ratio=tc["warmup_ratio"],          
            logging_steps=tc["logging_steps"],
            save_steps=tc["save_steps"],
            eval_steps=tc["eval_steps"],
            eval_strategy=tc["eval_strategy"],
            save_strategy=tc["save_strategy"],
            save_total_limit=tc["save_total_limit"],
            load_best_model_at_end=tc["load_best_model_at_end"],
            metric_for_best_model=tc["metric_for_best_model"],
            bf16=tc["bf16"],
            max_length=tc["max_length"],
            max_grad_norm=tc["max_grad_norm"],
            gradient_checkpointing=tc.get("gradient_checkpointing", False),
        )

    def build_trainer(self, train_dataset, eval_dataset):
        sft_config = self._build_sft_config()
        self.trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )
        return self.trainer

    def train(self):
        if self.trainer is None:
            raise RuntimeError("build_trainer() must be called before train().")
        return self.trainer.train()

    def run(self, train_dataset, eval_dataset):
        self.load_model_and_tokenizer()
        self.apply_lora()
        self.build_trainer(train_dataset, eval_dataset)
        return self.train()