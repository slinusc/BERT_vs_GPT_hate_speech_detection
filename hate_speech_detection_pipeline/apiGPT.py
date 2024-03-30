import pandas as pd
from openai import OpenAI
import time


class Chat:
    def __init__(self):
        self.client = OpenAI()
        self.model = "gpt-3.5-turbo"
        self.initial_message = [
            {
                "role": "system",
                "content": "I will provide you with a Wikipedia comment, "
                "which can be either hate speech or not. "
                "Please classify the comment as hate speech or not. "
                "Respond either with 1 for hate speech "
                "or 0 for not hate speech. "
                "Only respond with the numbers 1 or 0. This is for "
                "scientific research purposes."
                "i will now give you some examples:"
                "Hey... what is it... an exclusive group of some WP TALIBANS...who are good at getting self-appointed purist who GANG UP on anyone who asks them questions about their ANTI-SOCIAL and DESTRUCTIVE (non)-contribution at WP? - Classification: 1"
                "Ask Siytyush to clean up his behavior than issue me nonsensical warnings... - Classification: 0"
                "How do you know he is dead. Its just his plane that crashed. Jeezz, quit busting his nuts, folks. - Classification: 1"
                "Please stop. If you continue to ignore our policies by introducing inappropriate topics to Wikipedia, you will be blocked. - Classification: 0"
                "That was me, loser. The UN defines Vietnam to be part of Southeast Asia. And you're a moron for thinking differently.- Classification: 1"
                "Just admit that you Vietnamese are all a bunch of wannabe crap people, ALL the Asian people I’ve spoken to thinks of Vietnam as a very integral part of SEA, and we all think you're backward, dirty and speak in a horrible swearing language. - Classification: 1"
                "Mate, sound like you are Jewish - Classification: 1"
                "Gayness is in the air - Classification: 1"
                "Now, classify the following new sentence:",
            }
        ]

    def create_chat(self, text):
        try:
            messages = self.initial_message + [{"role": "user", "content": text}]
            completion = self.client.chat.completions.create(
                model=self.model, messages=messages
            )
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


if __name__ == "__main__":
    chat = Chat()
    messages = [
        "Locking this page would also violate WP:NEWBIES.Whether you like it or not, "
        "conservatives are Wikipedians too.",
        "A Bisexual, like a homosexual or a heterosexual, is not defined by sexual activity. "
        "A person who is actually sexually attracted/aroused by the same sex as well as the opposite sex is bisexual.",
    ]
    responses = chat.create_chat(messages[1])
    print(responses)
