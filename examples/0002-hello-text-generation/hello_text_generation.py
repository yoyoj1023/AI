from transformers import pipeline


generator = pipeline("text-generation")
result = generator(
    "In this course, we will teach you how to",
    num_return_sequences=2,
    max_length=30,
)

print(result)
