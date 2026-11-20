from transformers import pipeline


classifier = pipeline("sentiment-analysis")
result = classifier("Hello world!!")

print(result)
