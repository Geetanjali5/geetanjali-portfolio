"""
profile_data.py

This is the ONLY knowledge source for Geetanjali AI. Every chunk here is
grounded in the content already on the portfolio page. The RAG pipeline
retrieves from this list and the LLM is instructed to answer ONLY using
retrieved chunks — never invent facts that aren't here.

To update what the assistant knows, edit this list. No retraining needed —
chunks are re-embedded automatically the next time the server starts.

Last synced against Geetanjali's resume (Panipat, Haryana) on 2026-08-16.
"""

PROFILE_CHUNKS = [
    {
        "id": "about",
        "topic": "About Geetanjali",
        "text": (
            "Geetanjali is an aspiring AI/ML professional and B.E. student in "
            "Artificial Intelligence & Machine Learning (Hons.) at Khalsa College of "
            "Engineering and Technology, Amritsar, Punjab (2023–2027), with a CGPA of "
            "8.71. She focuses on building practical, deployable AI systems across "
            "Machine Learning, Generative AI, LLMs, RAG, NLP, and Computer Vision — "
            "taking ideas from data and experimentation through APIs into usable "
            "applications, not just notebooks. She is based in Panipat, Haryana, "
            "India, and is seeking AI/ML, Generative AI, or Data Science "
            "opportunities, including internships and entry-level roles, remote or "
            "relocation."
        ),
    },
    {
        "id": "skills_ml",
        "topic": "Machine Learning skills",
        "text": (
            "Core Machine Learning skills: Supervised Learning, Classification, "
            "Regression, Feature Engineering, Model Optimization, and Scikit-learn. "
            "Working knowledge of Unsupervised Learning, Model Evaluation, "
            "Cross-Validation, and Hyperparameter Tuning. Familiar with Statistics "
            "and DSA (Data Structures & Algorithms)."
        ),
    },
    {
        "id": "skills_dl",
        "topic": "Deep Learning skills",
        "text": (
            "Core Deep Learning skills: Neural Networks and CNNs (Convolutional "
            "Neural Networks). Working knowledge of TensorFlow, PyTorch, and Model "
            "Training. Familiar with Transfer Learning."
        ),
    },
    {
        "id": "skills_genai",
        "topic": "Generative AI skills",
        "text": (
            "Core Generative AI skills: LLMs (Large Language Models), Prompt "
            "Engineering, RAG (Retrieval-Augmented Generation), and Embeddings. "
            "Working knowledge of Vector Search, LangChain, ChromaDB, and Mistral / "
            "OpenAI APIs."
        ),
    },
    {
        "id": "skills_nlp",
        "topic": "NLP skills",
        "text": (
            "Core NLP skill: Text Classification. Working knowledge of Text "
            "Summarization, Semantic Search, Transformers, and Hugging Face. Familiar "
            "with NLTK."
        ),
    },
    {
        "id": "skills_cv",
        "topic": "Computer Vision skills",
        "text": (
            "Core Computer Vision skills: OpenCV and Object Detection. Working "
            "knowledge of MediaPipe and Real-Time Vision. Familiar with general "
            "Image Processing and YOLO."
        ),
    },
    {
        "id": "skills_dev",
        "topic": "AI application development skills",
        "text": (
            "Core AI application development skills: Python, FastAPI, and Flask. "
            "Working knowledge of REST APIs. Familiar with React, JavaScript/"
            "TypeScript, C++, HTML, and CSS."
        ),
    },
    {
        "id": "skills_data",
        "topic": "Data and database skills",
        "text": (
            "Core data skills: NumPy, Pandas, and SQL. Working knowledge of ChromaDB "
            "(vector database), EDA (Exploratory Data Analysis), Power BI, and "
            "Tableau. Familiar with Matplotlib and Seaborn for visualization."
        ),
    },
    {
        "id": "skills_tools",
        "topic": "Tools and workflow",
        "text": (
            "Core tool: Git / GitHub for version control. Working knowledge of VS "
            "Code, Jupyter Notebook, and Google Colab."
        ),
    },
    {
        "id": "project_coursemate",
        "topic": "Project: CourseMate AI",
        "text": (
            "CourseMate AI is an AI-powered study assistant built with RAG "
            "(Retrieval-Augmented Generation) for querying PDFs, notes, and "
            "assignments. Problem: students need fast, accurate answers grounded in "
            "their own course material rather than generic web answers. Approach: "
            "documents are ingested, split into chunks, and converted into "
            "embeddings; a semantic retrieval step pulls the most relevant chunks for "
            "a question, which are passed to an LLM to generate a context-aware, "
            "conversational answer. Results: 90%+ answer relevance, retrieval time "
            "reduced by 75%, and sub-2-second responses. Tech stack: Python, "
            "LangChain, RAG, ChromaDB, OpenAI Embeddings, Mistral LLM, and FastAPI."
        ),
    },
    {
        "id": "project_dentaladvisor",
        "topic": "Project: DentalAdvisor",
        "text": (
            "DentalAdvisor is a RAG-powered dental document intelligence and AI "
            "assistant. Problem: dental documents and patient information are dense "
            "and hard to search through manually. Approach: it processes documents, "
            "performs retrieval and semantic search over them, and uses an LLM to "
            "generate accurate, grounded answers, with a React frontend on top of the "
            "FastAPI backend. Result: streamlined vector indexing cut query latency "
            "by 70%+. Tech stack: Python, LangChain, FastAPI, Mistral AI, Hugging "
            "Face, and React."
        ),
    },
    {
        "id": "project_vision_assistant",
        "topic": "Project: Real-Time Assistance System for Visually Impaired",
        "text": (
            "A Real-Time Assistance System for Visually Impaired users, built as an "
            "IoT-based navigation assistant. Approach: real-time obstacle detection "
            "combining computer vision (OpenCV) with ultrasonic sensor fusion, plus "
            "speech-based feedback for hands-free navigation, packaged as a "
            "lightweight, energy-efficient prototype. Tech stack: Arduino, Python, "
            "OpenCV, and IoT."
        ),
    },
    {
        "id": "project_sign_language",
        "topic": "Project: AI-Powered Sign Language Interpreter",
        "text": (
            "An AI-Powered Sign Language Interpreter addressing the communication gap "
            "between deaf/hard-of-hearing and hearing communities. It's a real-time "
            "computer vision system using MediaPipe hand-landmark detection and "
            "Machine Learning for gesture recognition, integrated with Text-to-Speech "
            "(TTS) to convert sign language into text and speech. The inference "
            "pipeline was optimized for low latency and high accuracy. Tech stack: "
            "Python, OpenCV, MediaPipe, and Scikit-learn."
        ),
    },
    {
        "id": "project_fake_news",
        "topic": "Project: Fake News Detection System",
        "text": (
            "A Fake News Detection System for distinguishing real from fabricated "
            "news articles at scale. Approach: processed 10,000+ news articles using "
            "TF-IDF vectorization and a Logistic Regression classifier, deployed via "
            "Flask with a web interface for real-time predictions. Self-reported "
            "classification accuracy is around 92% (still being independently "
            "verified). Tech stack: Python, TF-IDF, Scikit-learn, NLTK, and Flask."
        ),
    },
    {
        "id": "project_text_summarizer",
        "topic": "Project: Text Summarizer",
        "text": (
            "A Text Summarizer project for condensing long text into concise, "
            "readable summaries, built with Python and NLP techniques. The full "
            "implementation is available on GitHub at "
            "github.com/Geetanjali5/text-summarizer-project."
        ),
    },
    {
        "id": "experience_solitaire",
        "topic": "Experience: Data Science Intern at Solitaire Infosys",
        "text": (
            "Geetanjali worked as a Data Science Intern at Solitaire Infosys Pvt. "
            "Ltd., Chandigarh, from June 2025 to August 2025. She conducted EDA and "
            "data preprocessing on 10+ datasets using Python, Pandas, and NumPy. She "
            "built and evaluated 4+ ML models (Regression, Decision Tree, Random "
            "Forest, and KNN) with feature engineering and model optimization, and "
            "created 5+ visualizations and dashboards using Matplotlib and Seaborn to "
            "communicate actionable insights. She applied hyperparameter tuning and "
            "5-fold cross-validation to improve model performance and "
            "generalization."
        ),
    },
    {
        "id": "education",
        "topic": "Education",
        "text": (
            "Geetanjali is pursuing a B.E. in Artificial Intelligence & Machine "
            "Learning (Hons.) at Khalsa College of Engineering and Technology, "
            "Amritsar, Punjab, from 2023 to 2027, with a CGPA of 8.71. Before that, "
            "she studied at Dr. Maharaja Krishan Kapoor Arya Model School, Panipat, "
            "Haryana, completing her Intermediate (CBSE) with 82% in 2021 and her "
            "Matriculation (CBSE) with 90% in 2023."
        ),
    },
    {
        "id": "certifications",
        "topic": "Certifications & Achievements",
        "text": (
            "Certifications and training Geetanjali has listed: Machine Learning "
            "Crash Course (Google) and AI & Data Science (NASBA), plus hands-on "
            "training in Advanced IoT Systems conducted by IIT Ropar."
        ),
    },
    {
        "id": "achievements",
        "topic": "Achievements",
        "text": (
            "Achievements: recognized as a Hackathon Winner for developing an "
            "innovative AI-based solution; maintaining an 8.71 CGPA in her B.E. AI & "
            "ML program; completed hands-on training in Advanced IoT Systems "
            "conducted by IIT Ropar; completed Google's Machine Learning Crash "
            "Course; completed the AI & Data Science program from NASBA; completed a "
            "Data Science internship at Solitaire Infosys Pvt. Ltd., Chandigarh; and "
            "organized technical events and workshops at Khalsa College of "
            "Engineering and Technology (KCET), Amritsar."
        ),
    },
    {
        "id": "contact",
        "topic": "Contact and links",
        "text": (
            "Geetanjali's GitHub is github.com/Geetanjali5 and her LinkedIn is "
            "linkedin.com/in/geetanjali-b41859313. Her email is "
            "geetanjalibhola28@gmail.com and her phone number is +91-79880-16559. "
            "She is based in Panipat, Haryana, India, and is open to remote work or "
            "relocation for AI/ML, Generative AI, or Data Science internships and "
            "entry-level roles."
        ),
    },
    {
        "id": "why_hire",
        "topic": "Why hire Geetanjali",
        "text": (
            "Geetanjali is an aspiring AI/ML professional with hands-on experience "
            "developing Machine Learning, Generative AI, LLMs, RAG, NLP, and "
            "Computer Vision solutions, proficient in Python, LangChain, FastAPI, "
            "TensorFlow, Scikit-learn, and OpenCV, with practical exposure to "
            "semantic search, vector databases, prompt engineering, embeddings, and "
            "REST API development. She has shipped multiple end-to-end AI "
            "applications — not just notebooks — spanning Generative AI/RAG "
            "assistants (CourseMate AI, DentalAdvisor) with measurable performance "
            "gains, computer vision for accessibility (a sign language interpreter "
            "and an IoT-based assistance system for the visually impaired), and "
            "classic ML/NLP systems (fake news detection). She has "
            "production-adjacent experience from a Data Science internship at "
            "Solitaire Infosys, working end-to-end from data preprocessing to model "
            "evaluation and deployment. She's seeking AI/ML, Generative AI, or Data "
            "Science opportunities where she can keep building scalable, real-world "
            "AI solutions."
        ),
    },
]