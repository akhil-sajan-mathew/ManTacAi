import sys
import re

file_path = 'c:/Users/HP/Desktop/ManTacAi/backend/main.py'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Add the import
import_stmt = 'from manipulation_detection.src.utils.action_handlers.narrative_generator import generate_narrative_summary\n'
if import_stmt not in text:
    text = text.replace('import json\n', 'import json\n' + import_stmt)

# 2. Add the endpoint
endpoint_code = '''
class NarrativeResponse(BaseModel):
    narrative: str

@app.post("/api/full-analysis", response_model=NarrativeResponse)
async def get_full_analysis(request: dict):
    narrative = generate_narrative_summary(request)
    return NarrativeResponse(narrative=narrative)

'''
if '@app.post("/api/full-analysis")' not in text:
    text = text.replace('@app.post("/api/reset")', endpoint_code + '@app.post("/api/reset")')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)
print('Patch applied successfully')
