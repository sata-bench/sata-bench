import re
import pandas as pd
import os

from satabench.methods.choice_funnel.sata_prompt_builder import SATAPromptBuilder


def prompt_provider(prompt_type: str):
    # Step 1: load raw SATA-Bench dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)),
                             "methods/data",
                             "sata_bench_final_2025.json")
    data = pd.read_json(data_path,
                        orient='records',
                        lines=True)

    sata_promptv1_list = []
    alphabet_list = [chr(i) for i in range(ord('A'), ord('Z') + 1)] 

    # Step 2: construct prompt using dataset
    # For our evaluation we intentionally choose minimal
    # prompt design to remove noise of prompting.
    for _, row in data.iterrows():
        paragraph = row['paragraph']
        question = row['question']
        answer = ''
        dataset = row['dataset']

        choices_map = {}
        for idx, item in enumerate(row['choices']):
            choice = item[0]
            choice_id = alphabet_list[idx]
            is_gt = item[1]
            if is_gt:
                answer += choice_id
            choices_map[choice_id] = choice
    
        # Prompt V0: basic prompt for SATA-Bench
        text_v0 = """You are presented with the following:
Paragraph: {paragraph}
Question: {question}
Choices:
{{choices}}

Task:
Identify and select all the correct answers based on the paragraph and the question.
{{additional_requirements}}
Answers:"""

        # Prompt V2: yes or no questions
        text_v1 = """You are presented with the following:
Paragraph: {paragraph}
Question: {question}
Statement: {{choice}}

Task:
Determine if the statement answers the question correctly and reply with "Yes" or "No" only.
{{additional_requirements}}
Answer:"""


        # populate the template with paragraph and question
        if prompt_type == "yesno":
            prompt_template = text_v1.format(paragraph=paragraph, question=question)
        else:
            # prompt for all other methods
            prompt_template = text_v0.format(paragraph=paragraph, question=question)

        prompt_template = re.sub(r"\{\s+\{\s*.*?\s*\}\s+\}", "", prompt_template)
        sata_promptv1 = SATAPromptBuilder(prompt_template, choices_map, answer, dataset)
        sata_promptv1_list.append(sata_promptv1)

    return sata_promptv1_list
