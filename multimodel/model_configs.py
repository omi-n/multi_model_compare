from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Optional
from langchain.callbacks.manager import CallbackManager
import os
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import LlamaCpp

openai_set = {"gpt4-32k", "gpt4-8k", "gpt35-turbo", "text-davinci"}
llama_set = {"llama-7b", "llama-13b", "llama-30b", "llama-65b"}
alpaca_set = {"alpaca-7b", "alpaca-30b","alpaca-64b"}

def get_model(model_name, max_tokens=256, callbacks=None, verbose=False, dotenv_path=None):
    from dotenv import load_dotenv
    if dotenv_path is not None:
        load_dotenv(dotenv_path)
    else:
        load_dotenv()
    
    proc_name = model_name.lower()
    if proc_name in openai_set:
        chat = False
        match proc_name:
            case "gpt4-32k":
                config = GPT432KConfig(callbacks=callbacks)
                chat = True
            case "gpt4-8k":
                config = GPT48KConfig(callbacks=callbacks)
                chat = True
            case "gpt35-turbo":
                config = GPT35TurboConfig(callbacks=callbacks)
                chat = True
            case "text-davinci":
                config = TextDavinciConfig(callbacks=callbacks)
            case _:
                raise NotImplementedError("Model not yet supported!")
            
        if chat:
            return AzureChatOpenAI(
                deployment_name=config.deployment_name,
                model_name=config.model_name,
                callbacks=config.callback_manager,
                max_tokens=max_tokens,
                verbose=verbose
            )
        else:
            return AzureOpenAI(
                deployment_name=config.deployment_name,
                model_name=config.model_name,
                callbacks=config.callback_manager,
                max_tokens=max_tokens,
                verbose=verbose
            )
    
    elif proc_name in llama_set or proc_name in alpaca_set:
        match proc_name:
            case "llama-7b":
                config = LLaMA7BConfig(callbacks=callbacks)
            case "llama-13b":
                config = LLaMA13BConfig(callbacks=callbacks)
            case "llama-30b":
                config = LLaMA30BConfig(callbacks=callbacks)
            case "llama-65b":
                config = LLaMA65BConfig(callbacks=callbacks)
            case "alpaca-7b":
                config = Alpaca7BConfig(callbacks=callbacks)
            case "alpaca-13b": # TODO: add support when weights available
                config = Alpaca13BConfig(callbacks=callbacks)
            case "alpaca-30b":
                config = Alpaca30BConfig(callbacks=callbacks)
            case "alpaca-65b":
                config = Alpaca65BConfig(callbacks=callbacks)
            case _:
                raise NotImplementedError("Model not yet supported!")
            
        return LlamaCpp(
            model_path=config.local_path,
            callback_manager=config.callback_manager,
            max_tokens=min(256, max_tokens),
            verbose=verbose
        )
        
    else:
        raise NotImplementedError
        

@dataclass
@dataclass_json
class ModelConfig:
    def __init__(self,
                 deployment_name: Optional[str] = None,
                 model_name: Optional[str] = None,
                 local_path: Optional[str] = None,
                 backend: Optional[str] = None,
                 callback_manager: Optional[CallbackManager] = None
                ):
        self.deployment_name = deployment_name
        self.model_name = model_name
        self.local_path = local_path
        self.backend = backend
        self.callback_manager = callback_manager
    
    
class GPT432KConfig(ModelConfig):
    def __init__(self, callbacks: Optional[CallbackManager] = None):
        super().__init__(
            deployment_name="gpt-4-32k",
            model_name="gpt-4-32k",
            callback_manager=callbacks
        )
        
class GPT48KConfig(ModelConfig):
    def __init__(self, callbacks: Optional[CallbackManager] = None):
        super().__init__(
            deployment_name="gpt-4",
            model_name="gpt-4",
            callback_manager=callbacks
        )
        
class GPT35TurboConfig(ModelConfig):
    def __init__(self, callbacks: Optional[CallbackManager] = None):
        super().__init__(
            deployment_name="gpt-35-turbo",
            model_name="gpt-35-turbo (version 0301)",
            callback_manager=callbacks
        )
        
class TextDavinciConfig(ModelConfig):
    def __init__(self, callbacks: Optional[CallbackManager] = None):
        super().__init__(
            deployment_name="text-davinci-003",
            model_name="text-davinci-003",
            callback_manager=callbacks
        )
        
class LLaMA7BConfig(ModelConfig):
    def __init__(self, callbacks: Optional[CallbackManager] = None):
        super().__init__(
            model_name="LLaMA-7B",
            local_path=os.environ["LLAMA_7B_PATH"],
            callback_manager=callbacks
        )
        
class LLaMA13BConfig(ModelConfig):
    def __init__(self, callbacks: Optional[CallbackManager] = None):
        super().__init__(
            model_name="LLaMA-13B",
            local_path=os.environ["LLAMA_13B_PATH"],
            callback_manager=callbacks
        )

class LLaMA30BConfig(ModelConfig):
    def __init__(self, callbacks: Optional[CallbackManager] = None):
        super().__init__(
            model_name="LLaMA-30B",
            local_path=os.environ["LLAMA_30B_PATH"],
            callback_manager=callbacks
        )
        
class LLaMA65BConfig(ModelConfig):
    def __init__(self, callbacks: Optional[CallbackManager] = None):
        super().__init__(
            model_name="LLaMA-65B",
            local_path=os.environ["LLAMA_65B_PATH"],
            callback_manager=callbacks
        )
        
class Alpaca7BConfig(ModelConfig):
    def __init__(self, callbacks: Optional[CallbackManager] = None):
        super().__init__(
            model_name="Alpaca-7B",
            local_path=os.environ["ALPACA_7B_PATH"],
            callback_manager=callbacks
        )
        
class Alpaca13BConfig(ModelConfig):
    def __init__(self, callbacks: Optional[CallbackManager] = None):
        super().__init__(
            model_name="Alpaca-13B",
            local_path=os.environ["ALPACA_13B_PATH"],
            callback_manager=callbacks
        )

class Alpaca30BConfig(ModelConfig):
    def __init__(self, callbacks: Optional[CallbackManager] = None):
        super().__init__(
            model_name="Alpaca-30B",
            local_path=os.environ["ALPACA_30B_PATH"],
            callback_manager=callbacks
        )
        
class Alpaca65BConfig(ModelConfig):
    def __init__(self, callbacks: Optional[CallbackManager] = None):
        super().__init__(
            model_name="Alpaca-65B",
            local_path=os.environ["ALPACA_65B_PATH"],
            callback_manager=callbacks
        )
