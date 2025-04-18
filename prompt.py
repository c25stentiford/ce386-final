"""This script evaluates an LLM prompt for processing text so that it can be used for the wttr.in API"""

from ollama import Client
from pydantic import BaseModel

LLM_MODEL: str = "deepseek-r1:14b"  # Change this to be the model you want
client: Client = Client(
    host="http://localhost:11434"  # Change this to be the URL of your LLM
)

class Place(BaseModel):
    place: str
    point: bool

def llm_parse_for_wttr(prompt: str):
    response = client.chat(
        messages=[
            {
                "role": "system",
                "content": '''
                Your purpose is to extract the intended location from the user's weather request. Take your time to ensure correct response.
                The `point` field of the JSON output specifies if the location is a specific point, such as:
                "Changi Airport", "Hakata Station", "the Eiffel Tower", etc. Places which are NOT
                points are things like "Bangkok", "Leavenworth County", "Guam", "Lichtenstein", "Hokkaido Prefecture", 
                "Chatuchak District", "Lopburi Province", "Buckley Space Force Base", and "Wichita, Kansas".
                Otherwise, if the given location is a city, administrative division, or other geograhic locality, set `point` to false.
                Ensure it is in a concise format, such as:
                "Paris", "Denver, Colorado", "Taoyuan, Taiwan", or "New York City".
                If the given location appears to be a three-letter airport code, leave it intact and set `point` to false.
                Examples of airport codes include: "FUK", "MUC", etc.
                '''
            },
            {"role": "user", "content": prompt},
        ],
        model="deepseek-r1:14b",
        format=Place.model_json_schema()
    )        
    place = Place.model_validate_json(response.message.content)
    
    if place.point and not (place.place.upper() == place.place and len(place.place) == 3):
        place.place = "~" + place.place
    
    return place.place.replace(" ", "+")


# Test cases
test_cases = [
    {"input": "What's the weather in Rio Rancho?", "expected": "Rio+Rancho"},
    {"input": "What's the weather in Taipei?", "expected": "Taipei"},
    {"input": "What's the weather at Changi Airport?", "expected": "~Changi+Airport"},
    {"input": "Get the weather at MCI.", "expected": "MCI"},
    {"input": "What's the weather at DMK?", "expected": "DMK"},
    {"input": "Tell me the weather in Lansing, Kansas.", "expected": "Lansing,+Kansas"},
    {"input": "What's the weather in Fukuoka Prefecture?", "expected": "Fukuoka+Prefecture"}
]

def run_tests(test_cases: list[dict[str, str]]):
    """run_tests iterates through a list of dictionaries,
    runs them against an LLM, and reports the results."""
    num_passed = 0

    for i, test in enumerate(test_cases, 1):
        raw_input = test["input"]
        expected_output = test["expected"]

        print(f"\nTest {i}: {raw_input}")
        try:
            result = llm_parse_for_wttr(raw_input).strip()
            expected = expected_output.strip()

            print("LLM Output  :", result)
            print("Expected    :", expected)

            if result == expected:
                print("✅ PASS")
                num_passed += 1
            else:
                print("❌ FAIL")

        except Exception as e:
            print("💥 ERROR:", e)

    print(f"\nSummary: {num_passed} / {len(test_cases)} tests passed.")
    
run_tests(test_cases=test_cases)