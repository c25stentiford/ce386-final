"""Passes a personal intro statement to an LLM.
The LLM produces valid JSON that could be ingested into a database to create a new user.

Modified from https://ollama.com/blog/structured-outputs
"""

from ollama import chat
import sys
from pydantic import BaseModel

class Place(BaseModel):
    place: str

response = chat(
    messages=[
        {
            "role": "system",
            "content": '''
            Your purpose is to extract the intended location from the user's weather request.
            If the location is a specific point, location, or landmark, prefix it with a tilde.
            Otherwise, if it is a city or geograhic locality, simply ensure it is in a concise format, such as:
            "Paris", "Denver, CO", or "Taoyuan, Taiwan." If it appears to be a
            three-letter airport code, leave it intact, such as: "FUK", "MUC", etc.
            '''
        },
        {"role": "user", "content": sys.argv[1]},
    ],
    model="deepseek-r1:14b",
    format=Place.model_json_schema()
)

place = Place.model_validate_json(response.message.content)
print(place)
