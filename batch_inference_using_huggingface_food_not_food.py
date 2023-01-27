"""
Script to run batch inference using HuggingFace's zero-shot text-classification model on Modal.

Based on the tutorial: https://modal.com/docs/guide/ex/batch_inference_using_huggingface 

Goal: filter a large Hugging Face dataset for food-related images (based on the text caption).
"""
import io

import modal

stub = modal.Stub(
    "batch-inference-using-huggingface-for-food-not-food",
    image=modal.Image.debian_slim().pip_install(
        "datasets",
        "matplotlib",
        "scikit-learn",
        "torch",
        "transformers",
        "pandas"
    ),
)

# Defining the prediction function
class FoodNotFood:
    def __enter__(self):
        from transformers import pipeline

        self.classifier_pipeline = pipeline("zero-shot-classification",
                                            model="facebook/bart-large-mnli")

    @stub.function(cpu=8, retries=3)
    def predict(self, sample: dict, labels=["food", "not_food"]):
        # returns dict {"sequence": str, "labels": List[str], "scores": List[float]}
        output = self.classifier_pipeline(sample["top_caption"], labels)

        # Update sample with labels
        sample["prob_food"] = output["scores"][0]
        sample["prob_not_food"] = output["scores"][1]

        return sample

## Getting data
@stub.function
def get_data():
    from datasets import load_dataset

    dataset = load_dataset("laion/laion-coco", 
                           split="train", 
                           streaming=True)  # whole dataset is ~250GB so stream instead of downloading, see here: https://huggingface.co/datasets/laion/laion-coco/tree/main 

    dataset = dataset.remove_columns(["all_captions", "all_similarities", "WIDTH", "HEIGHT", "similarity", "hash"])

    # Shuffle the dataset and get 100 samples (to experiment with)
    shuffled_dataset = dataset.shuffle(buffer_size=100).take(100)
    return shuffled_dataset

@stub.local_entrypoint
def main():
    print("Downloading data...")
    data = get_data.call() # this tries to stream by default
    samples = list(data)
    try:
        print("Got", len(data), "reviews")
    except Exception as e:
        print("Exception while getting length of data:", e)


    # Let's check that the model works by classifying the first 5 entries
    predictor = FoodNotFood()
    for sample in samples[:5]:
        text = sample["top_caption"]
        prediction = predictor.predict.call(sample)
        print(f"Sample prob food: {prediction['prob_food']}: | prob not food: {prediction['prob_not_food']}")
        print(f"\nText:\n{text}\n\n")

    # Now, let's run batch inference over it
    print("Running batch prediction...")
    predictions = list(predictor.predict.map(samples))

    # Create a DataFrame of the samples
    import pandas as pd
    df = pd.DataFrame(predictions)
    
    # Save DataFrame to file
    df.to_csv("./predictions.csv", index=False)
    print(f"Wrote predictions to ./predictions.csv")

    
# Every container downloads the model when it starts, which is a bit inefficient.
# In order to improve this, what you could do is to set up a shared volume that gets
# mounted to each container.
# See [shared volumes](/docs/guide/shared-volumes).
#
# In order for Huggingface to use the shared volume, you need to set the value of
# the `TRANSFORMERS_CACHE` environment variable to the path of the shared volume.
# See [secrets](/docs/guide/secrets).