from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch


class DistilBERTModelClassificator:
    def __init__(self, model_path):
        """
        Initialisiert die ModelPredictor Klasse mit dem angegebenen Modell.

        Args:
        model_path (str): Pfad zur Modell-Checkpoint-Datei.
        """
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device)))
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def predict_label(self, text):
        """
        Macht eine Vorhersage für den übergebenen Text und gibt das vorhergesagte Label zurück.

        Args:
        text (str): Der Text, für den das Label vorhergesagt werden soll.

        Returns:
        int: Das vorhergesagte Label für den Text.
        """
        # Tokenize den Text
        tokenized_data = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        input_ids, attention_mask = tokenized_data['input_ids'].to(self.device), tokenized_data['attention_mask'].to(self.device)

        # Vorhersage machen
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, dim=1)

        return predicted.item()

if __name__ == "__main__":
    model = DistilBERTModelClassificator("model/model_distil.pth")
    print(model.predict_label("I hate you!"))
