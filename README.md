# IR Final Project — Wikipedia Search Engine (Pair Submission)

This repository contains the implementation of a Wikipedia search engine developed as part of the Information Retrieval final project.
The system is deployed on Google Cloud and exposes RESTful endpoints for searching Wikipedia articles using multiple retrieval strategies.

Live engine URL: http://35.184.171.107:8080  
GCS bucket: gs://shira_bucket_323014209  
GitHub repository: https://github.com/shiraat-1004/IR_Final_Project

---

## Project Structure

.
├── search_frontend.py        # Flask application: query processing, ranking logic, REST endpoints
├── inverted_index_gcp.py     # Inverted index implementation and GCS-based posting list readers
├── bench_queries.py          # Evaluation script for running benchmark queries and computing metrics
├── queries_train.json        # Benchmark queries with relevance judgments (provided dataset)
├── startup_script_gcp.sh     # Startup script for Google Cloud VM deployment
├── IR_Final_Project_Report.pdf  # Final project report (PDF)
├── README.md                 # Project documentation

---

## API Endpoints

All endpoints return results in JSON format.

GET /health  
Health check endpoint.  
Response: {"status":"ok"}

GET /search_body?query=<query>  
Retrieves up to 100 documents using TF-IDF weighting and cosine similarity over article body text.

GET /search_title?query=<query>  
Retrieves all documents that contain at least one query term in the article title.  
Documents are ranked by the number of distinct query terms matched in the title.

GET /search_anchor?query=<query>  
Retrieves all documents that contain at least one query term in anchor text.  
Documents are ranked by the number of distinct query terms matched in anchor text.

GET /search?query=<query>  
Combined retrieval strategy.  
Uses body-based TF-IDF cosine similarity as the primary signal and integrates additional title and anchor matching signals.

---

## Retrieval Methods

The system implements four main retrieval strategies:

Title-based retrieval  
Exact term matching over article titles. Fast but limited to entity-centric queries.

Body-based retrieval  
TF-IDF representation of the query and documents, ranked using cosine similarity.
This method provides the strongest baseline performance.

Anchor-based retrieval  
Uses anchor text pointing to articles. Useful in some cases but slower due to large posting lists.

Combined retrieval  
Blends body-based ranking with title and anchor signals to improve robustness while maintaining reasonable latency.
This is the final version submitted for evaluation.

---

## Tokenization and Stopwords

Tokenization follows the course specifications:

Regex-based tokenizer: [#@\\w](['\\-]?\\w){2,24}  
Lowercasing of all tokens  
Stopword removal using:
- NLTK English stopwords
- Additional Wikipedia-style corpus stopwords  
No stemming is applied.

---

## Configuration (Environment Variables)

The frontend supports configuration through environment variables.
Defaults are provided in the code.

Common variables:

IR_BUCKET_NAME  
Name of the Google Cloud Storage bucket containing all index files  
Example: shira_bucket_323014209

IR_BODY_DIR  
Directory or prefix of body index files in the bucket

IR_TITLE_DIR  
Directory or prefix of title index files in the bucket

IR_ANCHOR_DIR  
Directory or prefix of anchor index files in the bucket

Additional configuration variables can be found directly in search_frontend.py by searching for os.getenv.

---

## Running Locally

Create and activate a virtual environment:

python3 -m venv venv-ir  
source venv-ir/bin/activate  

Install dependencies:

pip install -U pip  
pip install flask nltk google-cloud-storage numpy  
python3 - <<EOF  
import nltk  
nltk.download('stopwords')  
EOF  

Run the server:

export IR_BUCKET_NAME=shira_bucket_323014209  
python3 -u search_frontend.py  

Test locally:

curl -s http://127.0.0.1:8080/health  
curl -s "http://127.0.0.1:8080/search?query=anarchism" | head  

---

## Deploying on Google Cloud VM

The system is deployed on a Google Cloud virtual machine with an external IP and an open firewall rule for TCP port 8080.

The provided startup script can be used during VM creation or run manually.

Manual deployment:

chmod +x startup_script_gcp.sh  
./startup_script_gcp.sh  

Run the server in the background:

pkill -f "python.*search_frontend" || true  
nohup python3 -u search_frontend.py > frontend.log 2>&1 &  

Verify from outside the VM:

curl -s http://35.184.171.107:8080/health  

---

## Evaluation

Evaluation was conducted using a fixed benchmark of 30 queries with relevance judgments.

Metrics:
Average Precision at 10 (AP@10)  
Precision at 5 (P@5)  
Recall at 30 (R@30)  
F1@30  
Average and percentile-based query latency  

The combined retrieval strategy achieved the best balance between effectiveness and efficiency.
Detailed results and graphs are provided in the final report.

---

## Caching Policy

Query results are not cached.  
Local caching of files downloaded from Google Cloud Storage is used to reduce repeated I/O.  
This behavior complies with the project requirements.

---

## Notes

This repository is intended for academic submission only.
All implementation choices follow the course guidelines and constraints.
