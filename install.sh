git clone https://github.com/huggingface/diffusers ./core/diffusers
pip install -U -r ./core/diffusers/examples/dreambooth/requirements.txt
pip install -r requirement.txt

python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install "paddleocr>=2.0.1"
python -m nltk.downloader stopwords

mkdir checkpoints
mkdir data
