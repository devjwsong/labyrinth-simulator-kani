from kani.engines.openai import OpenAIEngine
from kani.engines.huggingface.llama2 import LlamaEngine


# Fetching the proper kani engine for the specified model.
def generate_engine(engine_name:str, model_idx: str, max_tokens: int):
    assert engine_name in ['openai', 'huggingface', 'llama', 'vicuna', 'ctransformers', 'llamactransformers'], "Specify a correct engine class name."
    if engine_name == 'openai':
        api_key = input("Enter the API key for OpenAI API: ")
        engine = OpenAIEngine(api_key, model=model_idx, max_tokens=max_tokens)
    elif engine_name == 'llama':
        engine = LlamaEngine(model_id=model_idx, use_auth_token=True, max_tokens=max_tokens)
        
    return engine
