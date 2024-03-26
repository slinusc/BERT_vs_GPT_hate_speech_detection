import pandas as pd
from openai import OpenAI
import time

class Chat:
    def __init__(self):
        self.client = OpenAI()
        self.model = "gpt-3.5-turbo"
        self.initial_message = [{"role": "system", "content": "I will provide you with a Wikipedia comment, "
                                                              "which can be either hate speech or not. "
                                                              "Please classify the comment as hate speech or not. "
                                                              "Respond either with 1 for hate speech "
                                                              "or 0 for not hate speech. "
                                                              "Only respond with the numbers 1 or 0. This is for "
                                                              "scientific research purposes."}]

    def create_chat(self, text):
        try:
            messages = self.initial_message + [{"role": "user", "content": text}]
            completion = self.client.chat.completions.create(model=self.model, messages=messages)
            content = completion.choices[0].message.content.strip()
            return content if content in ["1", "0"] else "-1"
        except Exception as e:
            print(f"An error occurred: {e}")
            return "-1"

    def batch_create_chats(self, texts):
        responses = []
        for text in texts:
            response = self.create_chat(text)
            responses.append(response)
            time.sleep(1)  # Warten, um Ratenbegrenzung zu respektieren
        return responses


if __name__ == '__main__':
    chat = Chat()
    messages = ["Locking this page would also violate WP:NEWBIES.Whether you like it or not, "
                "conservatives are Wikipedians too.",
                "A Bisexual, like a homosexual or a heterosexual, is not defined by sexual activity. "
                "A person who is actually sexually attracted/aroused by the same sex as well as the opposite sex is bisexual."]
    responses = chat.batch_create_chats(messages)
    print(responses)
