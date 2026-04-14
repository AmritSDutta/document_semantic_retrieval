import requests
import time


def _get_messy_text():
    return """This is a test passage:
        - with "quotes"
        - with emojis 😃🔥
        - with unicode অ আ あ
        - with JSON-like {bad: 'json'}
        - with newlines and \t tabs
        - 😃🔥✨🚀📚🧠💡🔍🎯⚡📝🤖🌟💭🪄
        - এটা একটা সুন্দর বাংলা বাক্য।"""


def _get_messy_code():
    return """Need help with following code:
           for i, query in enumerate(queries):
        payload = {"search_term": query, "limit": 3}
        try:
            r = requests.post(url, json=payload, headers={"X-API-KEY": api_key})
            r.raise_for_status()

            # Extract process time from header
            proc_time = r.headers.get("x-process-time")
            if proc_time:
                latencies.append(float(proc_time))

            print(f"Req {i + 1} - Server Process Time: {proc_time}s")
        except requests.exceptions.RequestException as e:
            print(f"Request {i + 1} failed: {e}")

        if i < len(queries) - 1:
            time.sleep(1)
        """


queries = [
    _get_messy_text(),
    _get_messy_code(),
    "Search for a Blockchain Developer with specific experience in Hyperledger and Ethereum. Looking for candidates who have successfully built decentralized platforms for supply chain management in the international transportation sector. Must be proficient in writing smart contracts and optimizing logistics processes using SQL and Linux.",
    "Find a Business Analyst with a Certified Scrum Master (CSM) background and 7+ years of experience. Candidate should have a proven track record in automating warehouse management systems and warehouse logistics. Expertise in requirement gathering, Jira, and mapping workflow processes for global commerce is essential.",
    "Seeking a Senior Data Scientist with expertise in credit card fraud detection and stock sentiment analysis. The ideal candidate uses machine learning, NLP, and deep neural networks to find complex patterns. Must be proficient in Python, R, and deploying modeling algorithms to drive business impact.",
    "Search for a Database Administrator with 7+ years of hands-on experience in Oracle 12c and 19c. Candidates must have expertise in Data Guard for disaster recovery, RMAN for backup/recovery, and database migrations. Experience with SQL Server and performance tuning via AWR reports is highly preferred.",
    "Looking for a DevOps Engineer specialized in AWS cloud infrastructure and containerization. Candidate should be an expert in Terraform, Ansible, and Kubernetes. Must have experience building automated CI/CD pipelines using Jenkins and monitoring tools like Datadog or Splunk for large-scale production environments.",
    "Find a Senior DotNet Developer with expertise in .NET Core 6 and Azure services like Functions and Logic Apps. Candidate must have experience building single-page applications (SPA) using Blazor or Angular. Required skills include Entity Framework, REST API development, and SQL Server stored procedures.",
    "Seeking a Full Stack Java Developer with 6+ years of experience in Spring Boot and microservices architecture. Candidate should have handled AWS S3 data hosting and integrated Okta SSO for authentication. Proficiency in AngularJS, RESTful web services, and SQL is required for this role.",
    "Search for an ETL Developer or BI Developer with extensive experience in data warehousing and SSIS/SSRS. Candidate should be skilled in creating Tableau dashboards and performing data cleansing using Alteryx. Must have a strong background in SQL Server and optimizing complex T-SQL queries for reporting.",
    "Looking for an Epidemiologist or Data Scientist with specialized knowledge in medical reporting and patient data analysis. Candidate should be proficient in SPSS, R, and GIS. Experience in systematic reviews, meta-analysis, and using technology-based curriculum for instructional roles is a plus.",
    "Find a Digital Media Director with a background in social media strategy and content creation. Candidate should have experience managing marketing campaigns for radio stations or educational institutions. Skills must include SEO, SEM, Adobe Creative Suite, and managing high-volume digital content distribution."
]


def run_payload_pool():
    url = "http://localhost:8000/api/docs/search"
    api_key = "1234"
    latencies = []

    for i, query in enumerate(queries[:3]):
        payload = {"search_term": query, "limit": 3}
        try:
            r = requests.post(url, json=payload, headers={"X-API-KEY": api_key})
            r.raise_for_status()

            # Extract process time from header
            proc_time = r.headers.get("x-process-time")
            if proc_time:
                latencies.append(float(proc_time))

            print(f"Req {i + 1} - Server Process Time: {proc_time}s")
        except requests.exceptions.RequestException as e:
            print(f"Request {i + 1} failed: {e}")

        if i < len(queries) - 1:
            time.sleep(1)

    if latencies:
        avg_time = sum(latencies) / len(latencies)
        print(f"\n--- Statistics ---")
        print(f"Average x-process-time: {avg_time:.4f} seconds")


if __name__ == "__main__":
    run_payload_pool()
