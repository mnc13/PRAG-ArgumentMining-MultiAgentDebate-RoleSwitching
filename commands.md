cd framework
/.env & .env.example
set api key 

create gemini api key from google ai studio
create groq api key from groq

pip install -r requirements.txt
python main_pipeline.py

test run-
python run_eval_extended.py --offset 60 --limit 1 --runs 1


Window 1: python run_eval_extended.py --offset 0 --limit 10 --runs 3
Window 2: python run_eval_extended.py --offset 10 --limit 10 --runs 3
Window 3: python run_eval_extended.py --offset 20 --limit 10 --runs 3
Window 4: python run_eval_extended.py --offset 30 --limit 10 --runs 3
Window 5: python run_eval_extended.py --offset 40 --limit 10 --runs 3
Window 6: python run_eval_extended.py --offset 50 --limit 10 --runs 3


Window 1: python run_eval_extended.py --offset 60 --limit 10 --runs 3
Window 2: python run_eval_extended.py --offset 70 --limit 10 --runs 3
Window 3: python run_eval_extended.py --offset 80 --limit 10 --runs 3
Window 4: python run_eval_extended.py --offset 90 --limit 10 --runs 3
Window 5: python run_eval_extended.py --offset 100 --limit 10 --runs 3
Window 6: python run_eval_extended.py --offset 110 --limit 10 --runs 3
