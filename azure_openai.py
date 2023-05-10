from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import AzureChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from multimodel.model_configs import get_model
from tqdm import tqdm, trange

aoi_models = {
    "gpt3": {
        "deployment_name":"gpt-35-turbo",
        "model_name":"gpt-35-turbo (version 0301)", 
    }, 
    "gpt4": {
        "deployment_name": "gpt-4",
        "model_name": "gpt-4"
    }
}

template = """Question: {question}

Answer: Let's think step by step."""

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = get_model("gpt4-8k", dotenv_path="testing/.env", max_tokens=256)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What is the shortest possible number of moves in which a knight on an empty chessboard can move from d4 to g5? What about g5 to f6?"

for _ in trange(0, 3):
    with open("results/gpt4-validation.txt", "a") as f:
        chain_resp = question + "\n"
        chain_resp += llm_chain.run(question)
        chain_resp += "\n----------------------------------------------------------------\n"
        f.write(chain_resp)

