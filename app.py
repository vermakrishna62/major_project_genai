# Q&A Chatbot
#from langchain.llms import OpenAI

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import pandas as pd
import streamlit as st
import os
import pathlib
import textwrap
from PIL import Image
import PyPDF2 as pdf
import json

import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi


os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# =====================================================
# Below Code is for Youtube Transcript CODE
# =====================================================

# Function to extract transcript details from YouTube videos
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript

    except Exception as e:
        raise e

# Function to generate summary based on prompt from Google Gemini Pro
def generate_gemini_content(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text

# Function to display YouTube Transcribe Summarizer application
def show_youtube_transcribe_summarizer():
    st.header("YouTube Transcribe Summarizer Application")
    youtube_link = st.text_input("Enter YouTube Video Link:")

    if youtube_link:
        video_id = youtube_link.split("=")[1]
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

    if st.button("Get Detailed Notes"):
        transcript_text = extract_transcript_details(youtube_link)

        if transcript_text:
            prompt = """You are YouTube video summarizer. You will be taking the transcript text
                and summarizing the entire video and providing the important summary in points
                within 250 words. Please provide the summary of the text given here: """
            summary = generate_gemini_content(transcript_text, prompt)
            st.markdown("## Detailed Notes:")
            st.write(summary)

## Function to load OpenAI model and get respones


# ===================================================================================
# ATS Resume Scan code
# ===================================================================================

# Define the function to get the response for ATS resume scan
def get_ats_scan_result(input):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input)
    return response.text

def input_pdf_text(uploaded_file):
    reader=pdf.PdfReader(uploaded_file)
    text=""
    for page in range(len(reader.pages)):
        page=reader.pages[page]
        text+=str(page.extract_text())
    return text

# Define the ATS Resume Scan functionality
def show_ats_resume_scan():
    st.header("ATS Resume Scan")
    st.text("Improve Your Resume ATS")

    # Job Description input
    jd = st.text_area("Paste the Job Description")

    # Resume upload
    uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type="pdf", help="Please upload the PDF")

    # Submit button
    submit = st.button("Submit")

    # Process submission
    if submit:
        if uploaded_file is not None:
            # Extract text from the uploaded PDF resume
            resume_text = input_pdf_text(uploaded_file)

            # Generate input prompt for the Gemini model
             # Generate input prompt for the Gemini model
            input_prompt = f"""
                Hey! Act like a skilled or very experienced ATS (Application Tracking System)
                with a deep understanding of the tech field, software engineering, data science, data analysis,
                and big data engineering. Your task is to evaluate the resume based on the given job description.
                You must consider the job market is very competitive and you should provide the best assistance
                for improving the resumes. Assign the percentage matching based on JD and the missing keywords
                with high accuracy. 
                resume: {resume_text}
                description: {jd}
                I want the response in one single string having the structure
                {{"JD Match":"%","Missing Keywords":[],"Profile Summary":""}}
            """

            # Get response for ATS resume scan
            response = get_ats_scan_result(input_prompt)

            # Convert response to dictionary
            response_dict = json.loads(response)

            # Display the response in a table
            st.subheader("**__ATS Resume Scan Results:__**")
            st.write("**__JD Match:__** ", response_dict["JD Match"])
            st.write("**__Missing Keywords:__** ", ", ".join(response_dict["Missing Keywords"]))
            st.write("**__Profile Summary:__** ", response_dict["Profile Summary"])



##initialize our streamlit app

st.set_page_config(page_title="Gemini AI Toolkit")

# =============================================================
# Invoice Extractor Code
# =============================================================

def get_gemini_response(input,image,prompt):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([input,image[0],prompt])
    return response.text
    

def input_image_setup(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")


# Define the content of the "Multi Language Invoice Extractor" section
def show_invoice_extractor():
    st.header("Multi Language Invoice Extractor Application")
    input_data = st.text_input("Input Prompt: ", key="input")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    submit = st.button("Tell me about the image")

    if submit:
        image_data = input_image_setup(uploaded_file)
        input_prompt = """
                    You are an expert in understanding invoices.
                    We will upload an image as invoice and 
                    you will have to answer questions based on the uploaded invoice image
                    """
        response = get_gemini_response(input_prompt, image_data, input_data)
        st.subheader("The Response is")
        st.write(response)
        
# ====================================================================================
# Health App
# ====================================================================================

# Define the function to get the response from the Gemini model for the health app
def get_health_app_response(input, image):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([input, image[0]])
    return response.text

# Define the function to set up the input image for the health app
def input_image_setup_img(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Function to display the Health App
def show_health_app():
    st.header("Nutri Scan")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image = ""   
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

    submit_button = st.button("Tell me the total calories")

    input_prompt = """
    You are an expert in nutritionist where you need to see the food items from the image
    and calculate the total calories, also provide the details of every food item with calorie intake
    in the below format:

    1. Item 1 - no of calories
    2. Item 2 - no of calories
    ----
    ----
    Finally you can also mention whether the food is health or not and also mention the percentage split of the ratio of the carbohydrates,fats,fibers,sugar and other important things required in our diet
    
    """

    if submit_button:
        image_data = input_image_setup_img(uploaded_file)
        response = get_health_app_response(input_prompt, image_data)
        st.subheader("The Response is")
        st.write(response)



# =================================================================================
# Select the application
# =================================================================================
selected_application = st.selectbox("Select an application", ["Multi Language Invoice Extractor", "YouTube Transcribe Summarizer", "ATS Resume Scan", "Health App"])

# Show the content based on the selected application
if selected_application == "Multi Language Invoice Extractor":
    show_invoice_extractor()
elif selected_application == "YouTube Transcribe Summarizer":
    show_youtube_transcribe_summarizer()
elif selected_application == "ATS Resume Scan":
    show_ats_resume_scan()
elif selected_application == "Health App":
    show_health_app()