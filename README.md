# Llama2-on-CPU-Machine  

### steps 1:

clone the repository

```bash
git clone https://github.com/Manirathinam21/Llama2-on-CPU-Machine.git
```

### step 2:

create a virtual environment

```bash
conda create -n cpullama python=3.8 -y
```

```bash
conda activate cpullama
```

```bash
pip install -r requirements.txt
```

### step 3:

To run the application execute following command

```bash
python app.py

```

### Download the quantize model from the  link provided in model folder and keep the model in the model directory:

```ini
## Download the Llama 2 Model:

llama-2-7b-chat.ggmlv3.q4_0.bin

## From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
```