from kani.engines import openai


# Fetching the proper kani engine for the specified model.
def generate_engine(engine_name:str, model_index: str):
    assert engine_name in ['openai', 'huggingface', 'llama', 'vicuna', 'ctransformers', 'llamactransformers'], "Specify a correct engine class name."
    if engine_name == 'openai':
        api_key = input("Enter the API key for OpenAI API: ")
        engine = openai.OpenAIEngine(api_key, model=model_index)
        
    return engine
