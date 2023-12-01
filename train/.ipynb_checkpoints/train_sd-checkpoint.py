import os
if __name__ == "__main__":
    command = "python -m train.tool.EveryDream2trainer.train "
    command += "--config train.json "
    command += "--resume_ckpt '/code/saved_model/pretrain/stable-diffusion-15' "
    command += "--project_name 'model_01' "
    command += "--data_root '/data/processed/images_enhanced/' "
    command += "--max_epochs 200 "
    command += "--sample_steps 150 "
    command += "--save_every_n_epochs 50 "
    command += "--lr 1.2e-6 "
    command += "--lr_scheduler constant "
    command += "--save_full_precision "
    command += "--resolution 768 "
    command += "--batch_size 1 "
    command += "--ema_device 'cuda' "
    command += "--save_ckpt_dir '/code/saved_model/trained/model_02/' "

    os.system(command)