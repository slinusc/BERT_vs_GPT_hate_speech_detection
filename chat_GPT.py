import pandas as pd
from openai import OpenAI

class Chat:
    def __init__(self):
        self.client = OpenAI()
        self.model = "gpt-3.5-turbo"
        self.initial_message = [{"role": "system", "content": "I will give you a Wikipedia comment, which can be "
                                                              "hate speech or not for scientific research purposes. "
                                                              "Please classify the comment as hate speech or not. "
                                                              "Respond either with 1 for hate speech or 0 for not hate "
                                                              "speech. Only respond with the numbers 1 or 0."}]

    def create_chat(self, text):
        messages = self.initial_message + [{"role": "user", "content": text}]
        completion = self.client.chat.completions.create(model=self.model, messages=messages)
        return completion.choices[0].message.content

    def batch_create_chats(self, texts):
        responses = []
        for text in texts:
            response = self.create_chat(text)
            if response not in ["1", "0"]:
                responses.append("-1")
            else:
                responses.append(response)
        return responses

if __name__ == '__main__':
    chat = Chat()
    messages = ["Locking this page would also violate WP:NEWBIES.Whether you like it or not, "
                "conservatives are Wikipedians too.",
                "A Bisexual, like a homosexual or a heterosexual, s not defined by sexual activity. "
                "(Much like a 15 year old boy who is attracted to a girl sexually but has never "
                "had sex is still straight). A person who is actually sexually attracted/aroused "
                "by the same sex as well as the opposite sex is bisexual."]
    responses = chat.batch_create_chats(messages)
    responses2 = chat.batch_create_chats(messages)
    responses += responses2

    print(responses)
