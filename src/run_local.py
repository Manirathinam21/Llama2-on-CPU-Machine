from langchain import PromptTemplate
from langchain  import LLMChain
from langchain.llms import CTransformers
from src.helper import *


B_INST, E_INST= "[INST]", "[/INST]" 
B_SYS, E_SYS= "<<SYS>>\n", "\n<</SYS>>\n\n" 

instructions= "Convert the following text from English to Tamil: \n\n {text}"

SYSTEM_PROMPT= B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS
template = B_INST + SYSTEM_PROMPT + instructions + E_INST

prompt= PromptTemplate(input_variables=['text'], template= template)

llm=  CTransformers(model= 'model/llama-2-7b-chat.ggmlv3.q4_0.bin',
                    model_type= 'llama',
                    config={'max_new_tokens': 128,
                            'temperature': 0.01}
                    )

chain= LLMChain(llm=llm, prompt=prompt)

response=  chain.run('how are you?')
print(response)