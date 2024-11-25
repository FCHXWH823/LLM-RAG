import pandas as pd
import csv

if __name__ == '__main__':
    df_llm_response = pd.read_csv("eval-results.csv")
    df_rag_llm_response = pd.read_csv("Langchain-RAG-eval-results.csv")

    with open("assertion-generation-eval-results.csv","w") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['HumanExplanation','pure code','prompt','code','llm_response','llm_rag_response'])
        for i in range(len(df_llm_response)):
            humanexplanation = df_llm_response.iloc[i]['HumanExplanation']
            purecode = df_llm_response.iloc[i]['pure code']
            prompt = df_llm_response.iloc[i]['prompt']
            code = df_llm_response.iloc[i]['code']
            llm_response = df_llm_response.iloc[i]['llm_response']
            llm_rag_response = df_rag_llm_response.iloc[i]['llm_response']
            csv_writer.writerow([humanexplanation,purecode,prompt,code,llm_response,llm_rag_response])
    