import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GemmaTokenizer
#from google.colab import userdata

def main():
    print("is cuda torch av? : " + torch.cuda.is_available().__str__())
    os.environ["WANDB_DISABLED"] = "true"
    tokenizer, model = load_model()
    test_model(tokenizer, model)



def load_model():
    #os.environ["HF_TOKEN"] = userdata.get('HUGGINGFACEHUB_API_TOKEN')
    print("Loading model...")
    
    model_id = "google/gemma-7b"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # from web
    # tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
    # model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, 
    #                                              device_map={"":0}, token=os.environ['HF_TOKEN'])
    
    # from local
    tokenizer = AutoTokenizer.from_pretrained("./model")
    model = AutoModelForCausalLM.from_pretrained("./model", quantization_config=bnb_config, 
                                                 device_map={"":0})
    

    return tokenizer, model
    

def test_model(tokenizer, model):
    text = "Quote: Imagination is more"
    device = "cuda:0"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

from peft import LoraConfig
def GetLoraConfig():
    lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",)

    return lora_config



if __name__ == "__main__":
    main()


# https://dwin.tistory.com/155