import os
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
from src.mcqgenerator.logger import logging
from src.mcqgenerator.utils import read_file,get_table_data

# imorting necessary packages from langchain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI

#if we want our environment to work locally so we need to use this
load_dotenv()

# access the environment variabel just like you do os.environ
key=os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(openai_api_key=key, model_name="gpt-3.5-turbo",temperature=0.5)



#few shot prompt
template= """
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to\
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide.\
Ensure to make {number} MCQs.
### RESPONSE_JSON
{response_json}
"""

quiz_generation_prompt=PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template = template
)
quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)




template2 = """
You are an expert English grammarian and writer. Given a multiple choice quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for the complexity,
if the quiz is not as per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the students ability.
Quiz_MCQs:
{quiz}


Check from an expert English Writer of the above quiz:
"""

quiz_review_generation= PromptTemplate(
    input_variables=["subject","quiz"],
    template = template2
)

review_chain = LLMChain(llm=llm, prompt=quiz_review_generation, output_key="review", verbose=True)

generate_evaluate_chain=SequentialChain(
    chains=[quiz_chain,review_chain],
    input_variables=["text","number","subject","tone","response_json"],
    output_variable=["quiz","review"],
    verbose = True
    )