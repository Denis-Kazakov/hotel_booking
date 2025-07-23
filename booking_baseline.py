# Baseline booking "package"
# Same functions as in the booking_baseline notbook, except the booking agent now returns responses from both the first and the secont call to LLM

import json
from typing import Callable
from datetime import date
import logging
import requests

from ollama import ChatResponse, chat
from pydantic import BaseModel, Field

from data_models import HotelData, AvailabilityRequest, AvailabilityResponse, BookingRequest, BookingResponse


logging.basicConfig(level=logging.ERROR) 

def get_hotels(city: str, checkin_date: date, checkout_date: date):
    """
    Get data on all hotels available in a given city on given dates.
    
    Args:
        city: City where available hotels should be found
        checkin_date: Date when the user wants to check in (format: date from the datetime package)
        checkout_date: Date when the user wants to check out (format: date from the datetime package)
    
    Returns:
        JSON with the following keys:
            success: a Boolean with value True if there are hotels with rooms available for booking in the given city on the given dates and False if no hotel rooms are available for booking
            error_message: A string giving reasons, if any, for not fulfilling the request, e.g. city not found or no free rooms on the given dates or None if the search was successfull
            available_hotels: a list where each element is a dictionary with the following keys:
                name: hotel name
                star_rating: star rating of the hotel (an integer between 1 and 5)
                price: room price per night
        
    """
    api_url = 'http://127.0.0.1:8000/get_hotels'
    headers = {"Content-Type": "application/json"}
    response = requests.post(api_url,
                         headers=headers,
                         json={'city': city, 'checkin_date': checkin_date, 'checkout_date': checkout_date})
    logging.debug(f'Response from the get_hotel endpoint: {response}')
    return response.json()

def book_hotel(name: str, city: str, checkin_date: date, checkout_date: date):
    """
    Book a hotel room as per the user request.
    
    Args:
        name: name of the hotel where a room should be booked
        city: city the hotel is located
        checkin_date: date when the user wants to check in
        checkout_date: date when the user wants to check out
    
    Returns:
        JSON with the following keys:
            success: a Boolean with value True if booking was successful and False if not
            message: string describing the outcome, e.g. confirming the booking or giving reasons for failure
    """
    api_url = 'http://127.0.0.1:8000/book'
    headers = {"Content-Type": "application/json"}
    response = requests.post(api_url,
                         headers=headers,
                         json={'city': 'Moscow', 'name': name, 'checkin_date': checkin_date, 'checkout_date': checkout_date})
    return response.json()

available_functions = {
    "get_hotels": get_hotels,
    "book_hotel": book_hotel
}

class FunctionCall(BaseModel):
    """Single function call to be returned by an LLM"""
    name: str = Field(..., description='Name of the function to be called')
    arguments: dict[str, str | None] = Field(..., description='Arguments of the function to be called')

class FunctionCalls(BaseModel):
    """All function calls to be returned by an LLM to process a user's request"""
    function_calls: list[FunctionCall] = Field(default=[], description='List of function calls returned by an LLM')
    comment: str | None = Field(..., description='LLM comments on the data provided, for example pointing out to missing arguments')

class BookingAgent:
    def __init__(self, functions: dict[str, Callable], model_name: str, system_message_1: str, system_message_2: str):
        self.functions = functions
        self.model_name = model_name
        self.system_message_1 = system_message_1
        self.system_message_2 = system_message_2

    def __call__(self, user_request: str):
        """Respond to user queries"""

        # First call to an LLM
        response: ChatResponse = chat(
            self.model_name,
            messages=[
                {'role': 'system', 'content': self.system_message_1},
                {'role': 'user', 'content': user_request}
            ],
            tools=list(self.functions.values()),
            stream=False,
            format=FunctionCalls.model_json_schema()
        )
        logging.info(f'Response from the first call to LLM: {response}')

        # Function calls
        llm_response = json.loads(response.message.content)
        logging.debug(f'Parsed llm response: {llm_response}')
        api_response = []
        assistant_message = None
        if llm_response.get('function_calls'):
            try:
                for function_call in llm_response['function_calls']:
                    func_name = function_call['name']
                    if func_name in self.functions:
                        api_response.extend(
                            [
                                f'Response from function {func_name}:',
                                str(self.functions[func_name](**function_call['arguments']))
                            ]
                        )
            except Exception as e:
                logging.error(f'Function calls failed with error: {str(e)}')
                api_response = [llm_response.get('comment')]
        else:
            api_response = [llm_response.get('comment')]
        logging.info(f'Response from the booking service API: {api_response}')
        
        # Second call to the LLM
        prompt = f'''
User request:
{user_request}
Response from the booking service API:
{'/n'.join(api_response)}
        '''
        logging.debug(f'Prompt:\n{prompt}')
        response: ChatResponse = chat(
            self.model_name,
            messages=[
                {'role': 'system', 'content': self.system_message_2},
                {'role': 'user', 'content': prompt}
            ],
            stream=False
        )
        logging.info(f'Response from the second call to LLM: {response}')
        return llm_response, response.message.content

system_message_1 = '''You are a booking agent helping users to find available hotels or to book a room in a specific hotel. You accomplish this by calling relevant functions.

**Crucially, you MUST ONLY use data provided DIRECTLY by the user.
DO NOT invent any data or make any assumptions. 
If the user does not provide all the necessary information, return a JSON with an empty list of function calls and an explanation of the missing data.**
 
If the user wants to book a room in a specific hotel, do not search for other options, just call the function for booking the room.

**Your response MUST be valid JSON conforming to the following structure:**
{
    "function_calls": [
        {
            "name":"function name",
            "arguments":{
                "arg1": "value1",
                "arg2": "value2"
            }
        }
    ],
    "comment": "Explanation of missing data or success."
}

If the user request does not include all required parameters for the function call, return JSON with the above format containing an empty list of function calls and a comment explaining which data are missing.

Example #1:
User query: "Book me a room in The Luxury Collection hotel, Guangzhou, from August 1, 2025 to August 7, 2025".
Response: {
    "function_calls": [{"name":"book_hotel", "arguments":{"city":"Guangzhou", name: "The Luxury Collection", "checkin_date":"2025-08-01", "checkout_date":"2025-08-07"}}],
    "comment": ""
}

Example #2:
User query: "Find me hotels in Tokio".
Response: {
    "function_calls": [],
    "comment": "Missing checkin and checkout date"
}

Example #3:
User query: "Which hotels are available in Seoul between August 1, 2025 and August 2, 2025".
Response: {
    "function_calls": [{"name":"get_hotels", "arguments":{"city":"Seoul", "checkin_date":"2025-08-01", "checkout_date":"2025-08-02"}}],
    "comment": ""
}


Example #4:
User query: "Find me a hotel in London after September 1, 2026".
Response: {
    "function_calls": [],
    "comment": "Missing checkout date"
}


'''
system_message_2 = '''You are a booking agent helping users to find available hotels or to book a room in a hotel.
The user request was forwarded to the booking service API and it returned a response.

**Crucially, you MUST ONLY use data provided IN THE API RESPONSE.  DO NOT invent any data or make any assumptions. If there is no response, inform the user that the service is not available.**

Answer questions and requests disrelated to your tasks by decribing the tasks you are suppposed to accomplish.
'''