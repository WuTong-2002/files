from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

print("Loading model...")
# 加载中文 NER 模型
model_name = "ckiplab/bert-base-chinese-ner"  # 专门针对中文的NER模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

print("Creating pipeline...")
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

text = "马云在杭州创立了阿里巴巴公司。"
print(f"Input text: {text}")

print("Running NER pipeline...")
results = ner_pipeline(text)
print(f"Results: {results}")
for entity in results:
    print(entity)
