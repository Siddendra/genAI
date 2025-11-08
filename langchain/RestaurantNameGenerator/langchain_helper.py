from secret_key import OPENAI_API_KEY
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Setup ---
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY.strip()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)
to_str = StrOutputParser()

# --- Prompts ---
prompt_template_name = PromptTemplate(
    input_variables=["cuisine"],
    template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this."
)

prompt_template_items = PromptTemplate(
    input_variables=["restaurant_name"],
    template="Suggest some menu items for {restaurant_name}. Return it as a comma separated string."
)

# --- Chains (LCEL style) ---
name_chain = prompt_template_name | llm | to_str
items_chain = prompt_template_items | llm | to_str

def generate_restaurant_name_and_items(cuisine: str):
    restaurant_name = name_chain.invoke({"cuisine": cuisine}).strip()
    menu_items = items_chain.invoke({"restaurant_name": restaurant_name}).strip()
    return {"restaurant_name": restaurant_name, "menu_items": menu_items}

if __name__ == "__main__":
    result = generate_restaurant_name_and_items("Italian")
    print(result)
