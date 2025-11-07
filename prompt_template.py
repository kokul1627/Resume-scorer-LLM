 # prompt = """You are an helpful assistant in evaluating and scoring the applicant's resume against the job description. \
#     The job description is solefully related to Natural Language Processing(NLP) specialist and you will be more \
#     vigilant in evaluating against the NLP related skills. 
#     For this task, you will be provided with a resume in pdf format and a job description as a text file. \
#     The resume and job description are delimited by triple quotes.
    
#     Please follow the below instructions for job description
#     {format_instruction1}

#     Please follow the below instructions for resume
#     {format_instruction2}

#     Evaluate and score the resume against the details from job description as per below instructions.
#     {format_instruction3}
#     Return only a pure JSON string surrounded by triple backticks (```)
    
#     Resume: ```{resume}```
#     Job Description: ```{job_description}```

# """


jd_prompt = """You are a helpful assistant tasked with extracting the information from given Job Description. 
You will be provided with a job description, delimited by triple quotes. This job description is solefully \
to recruit the applicant who works predominantely in Natural Language Processing(NLP) projects/tasks. 

Please extract the information from the Job Description as per the following instructions:
{format_instruction}

Return the evaluation in a JSON format surrounded by triple backticks (```).

Job Description: ```{job_description}```"""

resume_prompt = """You are a helpful assistant tasked with extracting the information from given Resume. 
You will be provided with a resume, delimited by triple quotes and will be more vigilant in extracting \
the NLP related technical skills and supportive skills while extracting other techincal skills. 

Please extract the information from the Resume as per the following instructions:
{format_instruction}

Return the evaluation in a JSON format surrounded by triple backticks (```).

Resume: ```{resume}```"""

score_prompt = """You are a helpful assistant tasked with evaluating and scoring a resume against a job \
description. For this task, you will be provided with two extracted information from job description \
and resume, delimited by triple quotes.

Parsed job description:
{parsed_jd}
Parsed applicant's resume:
{parsed_resume}

Please evaluate the parsed resume against the parsed job description as per the following scoring \
instructions and the scores should range between 0 and 10:

Scoring Instructions:
{format_instruction}

Return the evaluation in a JSON format surrounded by triple backticks (```).
"""