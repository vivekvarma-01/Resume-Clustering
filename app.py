import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
import io

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:bny@localhost/resumedatabase'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define a model for the clustering results matching the schema
class Resume(db.Model):
    __tablename__ = 'resume'
    filename = db.Column(db.String(200), primary_key=True)
    label = db.Column(db.Integer, nullable=False)
    resume_text = db.Column(db.Text, nullable=False)  # Store resume text for skill searching

# Load a pre-trained sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def generate_embeddings(texts):
    embeddings = model.encode(texts)
    return embeddings

def get_cluster_summary(texts, labels, num_clusters):
    summaries = {}
    for i in range(num_clusters):
        cluster_texts = [text for text, label in zip(texts, labels) if label == i]
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(cluster_texts)
        terms = vectorizer.get_feature_names_out()
        sum_words = X.sum(axis=0)
        word_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        sorted_words = sorted(word_freq, key=lambda x: x[1], reverse=True)
        top_words = [word for word, freq in sorted_words[:5]]  # Top 5 words
        summaries[i] = ' '.join(top_words)
    return summaries

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            return "No file uploaded.", 400

        # Read the CSV file
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        df = pd.read_csv(stream)

        if 'Resume' not in df.columns or 'Filename' not in df.columns:
            return "The uploaded CSV file does not contain the required columns 'Resume' and 'Filename'.", 400

        resumes = df['Resume'].tolist()
        file_names = df['Filename'].tolist()

        # Generate embeddings
        embeddings = generate_embeddings(resumes)

        # Perform clustering using Agglomerative Hierarchical Clustering
        num_clusters = 10  # You can make this dynamic based on user input
        clustering_model = AgglomerativeClustering(n_clusters=num_clusters)
        labels = clustering_model.fit_predict(embeddings)

        # Clear existing data in the table
        db.session.query(Resume).delete()
        db.session.commit()

        # Save clustering results to the database
        for label, file_name, resume_text in zip(labels, file_names, resumes):
            resume = Resume(filename=file_name, label=int(label), resume_text=resume_text)  # Add resume text
            db.session.add(resume)
        db.session.commit()

        # Retrieve and organize results from the database
        clusters = {}
        results = Resume.query.all()
        for result in results:
            if result.label not in clusters:
                clusters[result.label] = []
            clusters[result.label].append(result.filename)

        # Get cluster summaries
        cluster_summaries = get_cluster_summary(resumes, labels, num_clusters)

        # Sort clusters by their keys
        sorted_clusters = dict(sorted(clusters.items()))

        return render_template('result.html', clusters=sorted_clusters, summaries=cluster_summaries, num_clusters=num_clusters)

    return render_template('upload.html')

@app.route('/search', methods=['POST'])
def search():
    skill = request.form.get('skill')
    file = request.files.get('file')
    
    if not skill:
        return "No skill provided.", 400

    if not file:
        return "No file uploaded.", 400

    # Read the CSV file
    stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
    df = pd.read_csv(stream)

    if 'Resume' not in df.columns or 'Filename' not in df.columns:
        return "The uploaded CSV file does not contain the required columns 'Resume' and 'Filename'.", 400
    
    skill_results = []
    for _, row in df.iterrows():
        resume_text = row['Resume']
        filename = row['Filename']
        if skill.lower() in resume_text.lower():
            skill_results.append(filename)
    
    if not skill_results:
        return render_template('skill_search_results.html', skill=skill, results=skill_results, message="No resumes found for the skill.")

    return render_template('skill_search_results.html', skill=skill, results=skill_results, message=None)

if __name__ == '__main__':
    app.run(debug=True)
