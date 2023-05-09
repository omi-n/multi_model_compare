from dataclasses import dataclass
from dataclasses_json import dataclass_json
import numpy as np
from typing import Optional
from langchain.callbacks.manager import CallbackManager
import os

@dataclass_json
@dataclass
class MultiModelPrompterConfig:
    dotenv_path: str = ".env"
    gpt4_32k: bool = False
    gpt4_8k: bool = False
    gpt35_turbo: bool = False
    text_davinci: bool = False
    # bits: 1111 = (7B, 13B, 30B, 65B)
    llama: str = False
    # bits: 1111 = (7B, 13B, 30B, 65B)
    alpaca: str = False
    
    # will add this at a later date
    gpt4all_j: bool = False

class MultiModelPrompter:
    def __init__(self, config: MultiModelPrompterConfig):        
        self.config = config
        
        # using these model names because I can't load everything at once
        # there is a factory get_model which will give us the models at runtime
        self.model_names = []
        
        if self.config.gpt4_32k:
            self.model_names.append("gpt4-32k")
            
        if self.config.gpt4_8k:
            self.model_names.append("gpt4-8k")
            
        if self.config.gpt35_turbo:
            self.model_names.append("gpt35-turbo")
            
        if self.config.text_davinci:
            self.model_names.append("text-davinci")
        
        if self.config.llama:
            if len(self.config.llama) > 4:
                raise AttributeError("There are only 4 llama models! String should be bitmap of 0 | 1")
            
            bool_map = ["llama-7b", "llama-13b", "llama-30b", "llama-65b"]
            for i in range(len(bool_map)):
                if self.config.llama[i] == "1":
                    self.model_names.append(bool_map[i])
                    
        if self.config.alpaca:
            if len(self.config.alpaca) > 4:
                raise AttributeError("There are only 4 llama models! String should be bitmap of 0 | 1")
            
            bool_map = ["alpaca-7b", "alpaca-13b", "alpaca-30b", "alpaca-65b"]
            for i in range(len(bool_map)):
                if self.config.llama[i] == "1":
                    self.model_names.append(bool_map[i])
                    
        if self.config.gpt4all_j:
            raise NotImplementedError