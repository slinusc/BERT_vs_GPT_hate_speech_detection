from transformers import BertTokenizer, BertForSequenceClassification
import torch


class BERTModelClassificator:
    def __init__(self, model_path):
        """
        Initializes the ModelPredictor class with the specified model.

        Args:
        model_path (str): Path to the model checkpoint file.
        """
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Using BertForSequenceClassification
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        # Using BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def predict_label(self, text):
        """
        Makes a prediction for the given text and returns the predicted label.

        Args:
        text (str): The text for which to predict the label.

        Returns:
        int: The predicted label for the text.
        """
        # Tokenize the text
        tokenized_data = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        input_ids, attention_mask = tokenized_data['input_ids'].to(self.device), tokenized_data['attention_mask'].to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, dim=1)

        return predicted.item()

if __name__ == "__main__":
    # Make sure to update the file name if necessary
    model = BERTModelClassificator("model/model2.pth")
    print(model.predict_label("I hate you!"))