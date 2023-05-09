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
llm = get_model("text-davinci", dotenv_path="testing/.env", max_tokens=256)

llm_chain = LLMChain(prompt=prompt, llm=llm)

# d4 f6

# d4 g4
# d4 h7

# d4 h2 h2 f6
# d4 g5 g5 f6
question = "What is the shortest possible number of moves in which a knight on an empty chessboard can move from d4 to f6?"

for _ in trange(0, 3):
    with open("out_davinci_4moves_midboard.txt", "a") as f:
        chain_resp = question + "\n"
        chain_resp += llm_chain.run(question)
        chain_resp += "\n----------------------------------------------------------------\n"
        f.write(chain_resp)

