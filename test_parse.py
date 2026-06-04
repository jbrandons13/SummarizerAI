import json
import re

def _extract_json(response: str):
    response = response.strip()
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if not json_match:
        json_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        content = json_match.group(1)
    else:
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1:
            content = response[start:end+1]
        else:
            content = response
    content = re.sub(r',\s*([}\]])', r'\1', content)
    with open('failed_json_output.txt', 'w') as f:
        f.write(content)
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        content = re.sub(r'^[^{]*', '', content)
        content = re.sub(r'[^}]*$', '', content)
        with open('failed_json_output2.txt', 'w') as f:
            f.write(content)
        try:
            return json.loads(content)
        except Exception as e2:
            print("Failed", e2)
            
