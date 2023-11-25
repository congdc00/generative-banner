from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


model_name = "VietAI/envit5-translation"
tokenizer = AutoTokenizer.from_pretrained(model_name)  
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

inputs = ["vi: VietAI là tổ chức phi lợi nhuận với sứ mệnh ươm mầm tài năng về trí tuệ nhân tạo và xây dựng một cộng đồng các chuyên gia trong lĩnh vực trí tuệ nhân tạo đẳng cấp quốc tế tại Việt Nam."]

class Translater():
    model = None
    
    @staticmethod
    def __init_model():
        Translater.model = model.generate(tokenizer(inputs, return_tensors="pt", padding=True).input_ids.to('cuda'), max_length=512)
    
    @staticmethod 
    def vie2eng(vie_input):
        if Translater.model==None:
            Translater.__init_model()

        text_input = [f"vi: {vie_input}"]
        output = model.generate(tokenizer(text_input, return_tensors="pt", padding=True).input_ids.to('cuda'), max_length=512)
        text_resutl = tokenizer.batch_decode(output, skip_special_tokens=True)
        text_resutl = text_resutl[0].replace("en: ", "")
        return text_resutl


