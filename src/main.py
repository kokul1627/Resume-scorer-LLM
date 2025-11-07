from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser 
import streamlit as st
import fitz
import prompt_template
import pandas as pd
import sys
import os
import json
from typing import List, Union
from collections import OrderedDict
from utils import Utils
from io import StringIO
from dotenv import load_dotenv, find_dotenv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class JobDescriptionTemplate(BaseModel):
    """
    Model representing the structure of a job description.

    Attributes:
        job_title (str): Job title mentioned in the Job Description.
        job_description (str): Description of roles and responsibilities of the job.
        educational_qualification (str): Required educational qualifications.
        experience (float): Years of experience required.
        technical_skills_mand (List[str]): Mandatory technical skills.
        technical_skills_nlp (List[str]): NLP-related skills supportive to mandatory technical skills.
        technical_skills_other (List[str]): Other supportive technical skills outside the NLP domain.
    """
    # response = {'job_tiltle   }

    job_title: str = Field(description="Job title mentioned in the Job Description")
    job_description: str = Field(description="Description of roles and responsibilities of the job.")
    educational_qualification: str = Field(description="The required education qualification mentioned in the Job Description")
    experience: float = Field(description="Total number of years of experience mentioned in the Job Description")
    technical_skills_mand: List[str] = Field(description="Mandatory/high level technical skills mentioned in the Job Description")
    technical_skills_nlp: List[str] = Field(description="Traditional NLP related skills which are supportive to mandatory/high level technical skills provided in the Job Description. Avoid extracting skills or methodologies that integrate Generative AI (GenAI) components such as RAG, Langchain, LLM and so on. This is exclusion of soft skills")
    technical_skills_other: List[str] = Field(description="Skills which are supportive to mandatory/high level technical skills provided in the Job Description. These skills should be related to other than traditional NLP domain. For example: Generative AI, Object detection/classification, Langchain, OpenAI, Azure Function app and so on can be considered. This is exclusion of soft skills")

class ResumeTemplate(BaseModel):
    """
    Model representing the structure of a resume.

    Attributes:
        candidate_name (str): Applicant's name.
        job_role (str): Current job role.
        educational_qualification (List[str]): Educational qualifications.
        experience_ml (float): Experience in Data Science/ML/AI.
        experience_other (float): Experience in non-ML/AI domains.
        technical_skills_mand (List[str]): High-level technical skills.
        technical_skills_nlp (List[str]): NLP-related skills.
        technical_skills_other (List[str]): Other technical skills outside NLP.
    """

    candidate_name: str = Field(description="Name of the applicant in Resume")
    job_role: str = Field(description="Current job role of the applicant in Resume")
    educational_qualification: List[str] = Field(description="The education qualifications mentioned in the Resume. Inclusion of courses and certifications related to AI and ML field. Example: Masters in Computer Science, Bachelor of Engineering, any MS degree, PG Diploma in AIML, AZ900 exams and so on. Exclude schooling and any courses/certifications.")
    experience_ml: float = Field(description="Total number of years of experience mentioned in the Resume against the Data Science/Machine Learning/Artificial Intelligence field")
    experience_other: float = Field(description="Total number of years of experience mentioned in the Resume against other than Data Science/Machine Learning/Artificial Intelligence field, if applicant possess different domain experience. If there is no any experience, mention as 0.")    
    technical_skills_mand: List[str] = Field(description="High level technical skills related to mentioned in the Resume. For example: Azure, AWS, Python, Machine Learning, NLP, Deep Learning and so on are high level technical skills. Exclude the application or algorithms of these skills")
    technical_skills_nlp: List[str] = Field(description="Traditional NLP related skills which are supportive to high level technical skills. Avoid extracting skills or methodologies that integrate Generative AI (GenAI) components such as RAG, Langchain, LLM and so on. NER, spacy, NLTK, ANN, RNN, Transformers, Text classification, Text summarization and so on are the good examples.")
    technical_skills_other: List[str] = Field(description="Other skills which are supportive to high level technical/mandatory skills. These skills should be related to other than traditional NLP domain. For example: Generative AI, Object detection/classification, Langchain, OpenAI, Azure Function app, ML algorithms and so on can be considered.")

class ScoringTemplate(BaseModel):
    """
    Model representing scoring results for a resume.

    Attributes:
        job_role_score (List[Union[float, str]]): Score and justification for job role match.
        academic_score (List[Union[float, str]]): Score and justification for academic qualification match.
        experience_ml_score (List[Union[float, str]]): Score and justification for ML/AI experience match.
        technical_skills_mand_score (List[Union[float, str]]): Score and justification for mandatory skills match.
        technical_skills_nlp_score (List[Union[float, str]]): Score and justification for NLP skills match.
        technical_skills_other_score (List[Union[float, str]]): Score and justification for other technical skills match.
        resume_scoring (float): Weighted average score for the resume.
    """

    job_role_score: List[Union[float, str]] = Field(description="Evaluation and scoring of applicant's job role against the role from Job Description. Output should include score and justification/reason for the scoring.")
    academic_score: List[Union[float, str]] = Field(description="Evaluation and scoring of applicant's degree or equivalent course/certification against the education qualification from Job Description.Output should include score and justification/reason for the scoring.")
    experience_ml_score: List[Union[float, str]] = Field(description="Evaluation and scoring of applicant's Data Science/Machine Learning/Artificial Intelligence field experience against the experience required from Job Description. If applicant's experience is close to required experience from Job Description, give higher weightage. Example: If job description specifies as 4, then 2,3 or 5 are accepted. Output should include score and justification/reason for the scoring.")
    technical_skills_mand_score: List[Union[float, str]] = Field(description="Comparison of high level technical skills related to the mandatory skills from Job Description. Output should include score and justification/reason for the scoring.")
    technical_skills_nlp_score: List[Union[float, str]] = Field(description="Comparison of NLP domain realted skills which are supportive to high level technical skills from resume against the supportive technical skills from Job Description. The technical related soft skills are excluded such as model training, model evaluation, mentoring and so on. Output should include score and justification/reason for the scoring.")
    technical_skills_other_score: List[Union[float, str]] = Field(description="Comparison of other skills (exclusion of NLP skills) which are supportive to high level technical skills against the supportive technical skills from Job Description. The technical related soft skills are excluded such as model training, model evaluation, mentoring and so on. Output should include score and justification/reason for the scoring.")
    resume_scoring: float = Field(description="Weighted average score for the resume against the Job Description. The score should have one digit after the decimal point.")

class ResumeMatcher(Utils):
    """
    A class for matching resumes with job descriptions and generating scores.

    Attributes:
        api_key (str): API key for the language model.
        llm_model: Initialized language model for processing.
        jd_prompt_template (str): Template for job description prompt.
        resume_prompt_template (str): Template for resume prompt.
        scoring_prompt_template (str): Template for scoring prompt.
        jd_parser (JsonOutputParser): Parser for job description.
        resume_parser (JsonOutputParser): Parser for resume.
        output_parser (JsonOutputParser): Parser for scoring.
    """
    def __init__(self) -> None:
        """
        Initializes the ResumeMatcher by setting up environment variables and initializing parsers.
        """
        super().__init__()
        _ = load_dotenv(find_dotenv())
        # os.environ["LANGCHAIN_TRACING_V2"] = self.config_data['langchain_tracing_v2']
        # os.environ["LANGCHAIN_ENDPOINT"] = self.config_data['langchain_endpoint']
        # os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
        # os.environ["LANGCHAIN_PROJECT"] = self.config_data['langchain_project']
        self.api_key = "your_api_key_here"
        self.llm_model = init_chat_model(
            "gemini-2.5-flash",
            model_provider="google_genai",
            temperature=0,
            api_key=self.api_key)
        self.jd_prompt_template = prompt_template.jd_prompt
        self.resume_prompt_template = prompt_template.resume_prompt
        self.scoring_prompt_template = prompt_template.score_prompt
        self.jd_parser = JsonOutputParser(pydantic_object=JobDescriptionTemplate)
        self.resume_parser = JsonOutputParser(pydantic_object=ResumeTemplate)
        self.output_parser = JsonOutputParser(pydantic_object=ScoringTemplate)

    def create_prompt(self, 
                      prompt: str, 
                      input_variable: List[str], 
                      output_parser: JsonOutputParser) -> PromptTemplate:
        """
        Creates a prompt template for the language model.

        Args:
            prompt (str): The prompt string.
            input_variable (List[str]): Variables to be filled in the prompt.
            output_parser (JsonOutputParser): Parser to format the output.

        Returns:
            PromptTemplate: Configured prompt template.
        """
        prompt_template = PromptTemplate(
            template=prompt,
            input_variables=input_variable,
            partial_variables={"format_instruction": output_parser.get_format_instructions()}
        )
        return prompt_template
    
    def create_response(self, 
                        prompt_template: PromptTemplate, 
                        output_parser: JsonOutputParser, 
                        content: dict) -> dict:
        """
        Generates a response from the language model.

        Args:
            prompt_template (PromptTemplate): The prompt template.
            output_parser (JsonOutputParser): The parser for output.
            content (dict): The input content for the language model.

        Returns:
            dict: Parsed response from the model.
        """
        prompt = prompt_template
        chain = prompt | self.llm_model | output_parser
        output = chain.invoke(content)
        return output
    
    def get_response(self, 
                     jd_content: str, 
                     resume_content: str) -> tuple:
        """
        Processes job description and resume content to generate responses.

        Args:
            jd_content (str): Job description content.
            resume_content (str): Resume content.

        Returns:
            tuple: Parsed job description, resume, and scoring responses.
        """
        jd_prompt = self.create_prompt(self.jd_prompt_template, ["job_description"], self.jd_parser)
        jd_response = self.create_response(jd_prompt, self.jd_parser, {"job_description":jd_content})       

        resume_prompt = self.create_prompt(self.resume_prompt_template, ["resume"], self.resume_parser)
        resume_response = self.create_response(resume_prompt, self.resume_parser, {"resume":resume_content})

        score_prompt = self.create_prompt(self.scoring_prompt_template, ["parsed_jd", "parsed_resume"], self.output_parser)
        score_response = self.create_response(score_prompt, self.output_parser, {"parsed_jd":jd_response, "parsed_resume":resume_response})

        return jd_response, resume_response, score_response      

    def final_response(self, 
                       jd_content: str, 
                       resume_content: str) -> tuple:
        """
        Generates the final response by comparing job description and resume.

        Args:
            jd_content (str): Job description content.
            resume_content (str): Resume content.

        Returns:
            tuple: DataFrames with comparison and scoring details.
        """
        jd_response, resume_response, score_response = self.get_response(jd_content, resume_content)
        
        jd_response.pop("job_description")
        jd = {k:", ".join(v) if isinstance(v, List) else v for k, v in jd_response.items()}
        new_keys = ["Job Title", "Education Qualification", "Experience", "Mandatory Skills", "NLP Skills", "Other Technical Skills"]
        jd = dict(zip(new_keys, list(jd.values())))
        comparision_df = pd.DataFrame([jd]).T
        comparision_df.columns = ["Required"]

        applicant_name = resume_response.pop("candidate_name")
        resume = {k:", ".join(v) if isinstance(v, List) else v for k,v in resume_response.items()}
        resume["experience"] = f"Experience AIML: {resume['experience_ml']}, Experience other: {resume['experience_other']}"
        resume.pop("experience_ml")
        resume.pop("experience_other")

        new_dict = OrderedDict()

        for key, value in resume.items():
            new_dict[key] = value
            if key == 'educational_qualification':
                new_dict['experience'] = resume['experience']
        resume_list = [v for k,v in new_dict.items()]
        comparision_df["Applicant's Details"] = resume_list

        score = score_response.pop("resume_scoring")
        scores = [i[0] for i in [v for k,v in score_response.items()]]
        reason = [i[1] for i in [v for k,v in score_response.items()]]
        comparision_df["Scores"] = scores
        comparision_df["Justification"] = reason
        
        df2 = pd.DataFrame([{"Applicant's Name": applicant_name,
                            "Overall Scoring": str(score)}])
        score_df = df2.set_index("Applicant's Name")

        return comparision_df, score_df

if __name__ == "__main__":
    matcher = ResumeMatcher()
    st.title("Resume Scorer")
    with st.form("resume_matcher"):
         resume_file = st.file_uploader(label="Please upload the applicant's resumes", type=['pdf'])
         jd_file = st.file_uploader(label="Please upload the job description ",type=['txt'])
         btn = st.form_submit_button("Get Score")
            
    if resume_file and jd_file and btn:
        with st.spinner("Analysing the applicant's resume!!!"):
            doc = fitz.open(stream=resume_file.read(), filetype="pdf")
            resume = "\n\n".join([j.get_text() for j in doc])
            jd = StringIO(jd_file.getvalue().decode("utf-8")).read()
            comparision_df, score_df = matcher.final_response(jd, resume)
            st.subheader("Detail scores")
            st.dataframe(pd.DataFrame(comparision_df),width=1200)
            st.subheader("Overall score")
            st.dataframe(pd.DataFrame(score_df),width=1200)
