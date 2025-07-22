import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.Template import Template as Template
template = Template(categories = ['sc','wf','rt','ec'], save_excel=True)
template.download_all_categories()
template.save_dataset("datasets/template.csv")