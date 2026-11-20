from transformers import pipeline


classifier = pipeline("sentiment-analysis")
result = classifier("I am excited to start my AI journey!")

print(result)
