from functions import retrieve_answer
import os
os.chdir('/Users/an/Data_Engineering/projects/rag_llms')
response=retrieve_answer()
print(response.content)