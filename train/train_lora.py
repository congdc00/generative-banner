from accelerate.utils import write_basic_config
# write_basic_config()

import os

SCRIPT_PATH = "./core/diffusers/examples/dreambooth/train_dreambooth.py"
MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
INSTANCE_DIR="./data/train/target/images/"
CLASS_DIR = "./data/train/class/images/"
OUTPUT_DIR="./checkpoints/01/"

if __name__ == "__main__":
    command = f"python {SCRIPT_PATH} --pretrained_model_name_or_path={MODEL_NAME} --instance_data_dir={INSTANCE_DIR} --class_data_dir={CLASS_DIR} --output_dir={OUTPUT_DIR} "
    command += "--with_prior_preservation "
    command += "--prior_loss_weight=1.0 "
    command += "--instance_prompt='a banner vietnamese' "
    command += "--class_prompt='a banner'"
    command += "--resolution=1024"
    command += "--train_batch_size=1 "
    command += "--gradient_accumulation_steps=1 "
    command += "--gradient_checkpointing "
    command += "--use_8bit_adam "
    command += "--enable_xformers_memory_efficient_attention "
    command += "--set_grads_to_none "
    command += "--learning_rate=2e-6 "
    command += "--lr_scheduler='constant' "
    command += "--lr_warmup_steps=0 "
    command += "--num_class_images=40 "
    command += "--max_train_steps=800 "

    os.system(command)


