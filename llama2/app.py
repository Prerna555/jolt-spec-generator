import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

## Function To get response from LLAma 2 model

def getLLamaresponse(input_text,output_text,transformation_type):

    ### LLama2 model //pass path of downloaded quantised 7b llama 2 from huggingface
    llm=CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01})
    
    ## Prompt Template

    template="""
        Write a jolt spec for {transformation_type} transformations for given {input_text}
        to desired output {output_text}.
            """
    
    prompt=PromptTemplate(input_variables=["input_text","output_text",'transformation_type'],
                          template=template)
    
    ## Generate the ressponse from the LLama 2 model
    response=llm(prompt.format(input_text=input_text,output_text=output_text,transformation_type=transformation_type))
    print(response)
    return response






st.set_page_config(page_title="Generate Spec",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Generate Spec 🤖")

input_text=st.text_input("Enter the input text")
output_text=st.text_input("Enter the output text")
## creating to more columns for additonal 2 fields

transformation_type=st.selectbox('transformation_type',
                            ('json-json','json-xml'),index=0)
    
submit=st.button("Generate")

## Final response
if submit:
    st.write(getLLamaresponse(input_text,output_text,transformation_type))