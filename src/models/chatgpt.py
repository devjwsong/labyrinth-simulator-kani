import openai
import os

class ChatGPTModel:
    def __init__(self):
        self.model_name = "gpt-3.5-turbo"
        self.set_account()
    
    # For setting the OpenAI API account.
    def set_account(self):
        print("Setting OpenAI account information...")
        org = input("Enter the organization: ")
        openai.organization = org

        api_key = input("Enter the API key: ")
        openai.api_key = api_key
        os.environ['OPENAI_API_KEY']=openai.api_key
    
    # Default generation method.
    def generate_response(self, messages, **decoding_params):
        completion = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=decoding_params['temp'] if 'temp' in decoding_params else 0.7,
            max_tokens=decoding_params['max_tokens'] if 'max_tokens' in decoding_params else 128,
            top_p=decoding_params['top_p'] if 'top_p' in decoding_params else 0.8,
            frequency_penalty=decoding_params['frequency_penalty'] if 'frequency_penalty' in decoding_params else 2,
            presence_penalty=decoding_params['presence_penalty'] if 'presence_penalty' in decoding_params else 0,
        )

        return completion['choices'][0]['message']['content']
