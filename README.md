<h1>VisualCit Interactive Feedback Dashboard</h1>

<p>The feedback dashboard is a comprehensive tool designed to evaluate 
and scrutinize the efficiency of the data preparation pipeline within 
Visual Cit. It provides a streamlined approach to visualization, promotes a 
deeper understanding of the pipeline's performance, facilitates iterative
improvement and optimizing the pipeline configuration parameters.
</p>

<h2>Installation Procedure</h2>

#### Clone the repository
https://github.com/SiriBonthu/VisualCit-Feedback-Dashboard.git

#### Install dependencies
cd VisualCit-Feedback-Dashboard/
pip install -r requirements.txt

#### Run migrations
python manage.py migrate

#### Start the development server
python manage.py runserver

#### Access the dashboard
http://127.0.0.1:8000/